from src.utils.log_fileload import log_start_file, log_end_file
from src.config.constants import ACL_MAX_INHERITANCE_DEPTH
log_start_file(__file__)

# =============================================================================
# ACL Cypher clause generators
# =============================================================================
# Standalone functions that produce Cypher WHERE-block fragments enforcing
# READ permission for a given effective user.  They are designed to be inserted verbatim
# into query strings built in services.
#
# Convention
# ----------
# • Every generated block starts with "\n      AND (" so it can be appended
#   directly after an existing WHERE expression.
# • The caller is responsible for injecting the parameter $effectiveUserId (str, the
#   Neo4j User.id of the requesting user) into the query parameters dict.
# • When effective_user_id is None the service layer should skip ACL entirely; these
#   functions always assume ACL is active.
# =============================================================================


def acl_document_where_clause(doc_var: str) -> str:
    """Return a Cypher AND-block that enforces READ permission for $effectiveUserId on
    the Document node bound to `doc_var`.

    The block enforces:
    - Soft-delete exclusion: doc.isDeleted = false
    - Tenant-admin bypass via 'tenant-admin' Capability (checked first for early exit)
    - Direct GRANT:READ on the document via Group membership
    - Inherited GRANT:READ via ancestor Dossier up to ACL_MAX_INHERITANCE_DEPTH levels deep (with isProtected break and early termination)

    Optimizations:
    - Early tenant-admin check avoids expensive ACL traversals for admins
    - LIMIT 1 on dossier containment stops after first valid path (prevents unbounded expansion)
    - Pre-filter dossiers to non-protected nodes at match time
    - isDeleted check applied at match time for fast filtering

    Args:
        doc_var: The Cypher variable name for the Document node.

    Returns:
        A Cypher string fragment starting with "\\n      AND …"
    """
    return f"""
      AND {doc_var}.isDeleted = false
      AND (
        EXISTS {{ MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}}) }}
        OR EXISTS {{
          MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)
               -[:`GRANT:READ`]->({doc_var})
        }}
        OR (
          NOT {doc_var}.isProtected
          AND EXISTS {{
            MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)
                 -[:`GRANT:READ`]->(dos:Dossier {{isProtected: false}})
                 -[:CONTAINS*1..{ACL_MAX_INHERITANCE_DEPTH}]->({doc_var})
            LIMIT 1
          }}
        )
      )"""


def acl_ontology_where_clause(ontology_var: str) -> str:
    """Return a Cypher AND-block that enforces READ permission for $effectiveUserId on
    the Ontology node bound to `ontology_var`.

    Mirrors the logic in OntologyService.check_read_access_on_ontology:
    - SUPER_ADMIN bypass via 'tenant-admin' Capability
    - Direct GRANT:READ on the Ontology via Group membership
    - Inherited GRANT:READ via an ancestor reachable through
      [:CONTAINS|DEFINES*1..ACL_MAX_INHERITANCE_DEPTH] that is itself not protected

    Args:
        ontology_var: The Cypher variable name for the Ontology node.

    Returns:
        A Cypher string fragment starting with "\\n      AND ("
    """
    return f"""
      AND (
        EXISTS {{ MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}}) }}
        OR EXISTS {{
          MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)
               -[:`GRANT:READ`]->({ontology_var})
        }}
        OR (
          NOT coalesce({ontology_var}.isProtected, false)
          AND EXISTS {{
            MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)
                 -[:`GRANT:READ`]->(anc)
                 -[:CONTAINS|DEFINES*1..{ACL_MAX_INHERITANCE_DEPTH}]->({ontology_var})
            WHERE NOT coalesce(anc.isProtected, false)
          }}
        )
      )"""


def acl_ontology_read_access_query() -> str:
    """Return the complete Cypher query that checks GRANT:READ access for $effective_user_id on
    a specific Ontology node identified by $ontology_id.

    Parameters (must be passed by the caller):
        $effective_user_id   — Neo4j User.id of the requesting user
        $ontology_id — id property of the target Ontology node

    Returns:
        A Cypher query string that yields a single row with key 'has_access' (bool).
    """
    return f"""
        // Resolve caller's group ids
        MATCH (u:User:SystemAcl {{id: $effective_user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl)
        WITH collect(g.id) AS groupIds

        // Tenant-admin bypass
        WITH groupIds,
             EXISTS {{
                 MATCH (tg:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}})
                 WHERE tg.id IN groupIds
             }} AS isTenantAdmin

        MATCH (o:Ontology {{id: $ontology_id}})

        // Gather ancestor chain (stops at first protected ancestor)
        OPTIONAL MATCH aclPath = (o)<-[:CONTAINS|DEFINES*1..{ACL_MAX_INHERITANCE_DEPTH}]-(ancestor)
        WHERE NOT coalesce(o.isProtected, false)
        WITH groupIds, isTenantAdmin, o,
             ancestor,
             length(aclPath) AS aclDepth,
             coalesce(ancestor.isProtected, false) AS ancestorProtected
        ORDER BY o.id, aclDepth ASC
        WITH groupIds, isTenantAdmin, o,
             collect({{id: ancestor.id, protected: ancestorProtected}}) AS ancestorChain
        WITH groupIds, isTenantAdmin, o,
             reduce(s = {{ids: [], stop: false}}, x IN ancestorChain |
                 CASE WHEN s.stop THEN s
                      ELSE {{ids: s.ids + x.id, stop: x.protected}}
                 END
             ) AS aclScope

        RETURN isTenantAdmin
            OR EXISTS {{
                   MATCH (gd:Group:SystemAcl)-[gr:`GRANT:READ`]->(o)
                   WHERE gd.id IN groupIds
               }}
            OR EXISTS {{
                   MATCH (ga:Group:SystemAcl)-[ga_r:`GRANT:READ`]->(anc)
                   WHERE ga.id IN groupIds
                     AND anc.id IN aclScope.ids
               }} AS has_access
    """


def acl_ontology_permission_check_query(permission: str) -> str:
    """Return the complete ACL query for a permission on an Ontology node.

    Label shape mirrors acl_ontology_read_access_query on purpose: principals live on
    :User:SystemAcl / :Group:SystemAcl and the tenant-admin capability is reached through
    (:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {id:'tenant-admin'}).
    Matching bare :Role / :Capability here never matches the live schema, which silently pins
    isTenantAdmin to false and denies every administrator.
    """
    allowed_permissions = {"READ", "ADD", "UPDATE", "DELETE"}
    normalized_permission = permission.upper()
    if normalized_permission not in allowed_permissions:
        raise ValueError(f"Unsupported ontology ACL permission: {permission}")

    grant_relationship = f"GRANT:{normalized_permission}"
    return f"""
        MATCH (u:User:SystemAcl {{id: $effective_user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl)
        WITH collect(g.id) AS groupIds
        WITH groupIds,
             EXISTS {{
                 MATCH (tg:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}})
                 WHERE tg.id IN groupIds
             }} AS isTenantAdmin
        MATCH (o:Ontology {{id: $ontology_id}})
        WITH o, groupIds, isTenantAdmin,
             EXISTS {{
                 MATCH (directGroup:Group:SystemAcl)-[:`{grant_relationship}`]->(o)
                 WHERE directGroup.id IN groupIds
             }} AS directGrant
        OPTIONAL MATCH aclPath = (o)<-[:CONTAINS|DEFINES*1..{ACL_MAX_INHERITANCE_DEPTH}]-(ancestor)
        WHERE NOT coalesce(o.isProtected, false)
        WITH o, groupIds, isTenantAdmin, directGrant,
             ancestor, length(aclPath) AS aclDepth,
             coalesce(ancestor.isProtected, false) AS ancestorProtected
        ORDER BY aclDepth ASC
        WITH groupIds, isTenantAdmin, directGrant,
             collect({{id: ancestor.id, protected: ancestorProtected}}) AS ancestorChain
        WITH groupIds, isTenantAdmin, directGrant,
             reduce(s = {{ids: [], stop: false}}, x IN ancestorChain |
                 CASE WHEN s.stop THEN s
                      ELSE {{ids: s.ids + x.id, stop: x.protected}}
                 END
             ) AS aclScope
        RETURN isTenantAdmin
            OR directGrant
            OR EXISTS {{
                MATCH (ancestorGroup:Group:SystemAcl)-[:`{grant_relationship}`]->(ancestor)
                WHERE ancestorGroup.id IN groupIds
                  AND ancestor.id IN aclScope.ids
            }} AS has_access
    """


def acl_ontology_any_read_access_query() -> str:
    """Return the complete Cypher query that checks whether $effective_user_id has GRANT:READ
    on at least one Ontology node (tenant-admin bypass included).

    Parameters (must be passed by the caller):
        $effective_user_id — Neo4j User.id of the requesting user

    Returns:
        A Cypher query string that yields a single row with key 'has_any_access' (bool).
    """
    return """
        MATCH (u:User:SystemAcl {id: $effective_user_id})-[:MEMBER_OF]->(g:Group:SystemAcl)
        WITH collect(g.id) AS groupIds
        RETURN
          EXISTS {
            MATCH (tg:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {id: 'tenant-admin'})
            WHERE tg.id IN groupIds
          }
          OR EXISTS {
            MATCH (gd:Group:SystemAcl)-[:`GRANT:READ`]->(:Ontology)
            WHERE gd.id IN groupIds
          } AS has_any_access
    """


def acl_dossier_where_clause(dossier_var: str) -> str:
    """Return a Cypher AND-block that enforces READ permission for $effectiveUserId on
    the Dossier node bound to `dossier_var`.

    The block enforces:
    - Direct GRANT:READ on the dossier via Group membership
    - Inherited GRANT:READ via ancestor Dossier (with isProtected break)
    - SUPER_ADMIN bypass via 'tenant-admin' Capability

    Designed to be appended to an existing WHERE clause (starts with AND).
    The caller must pass $effectiveUserId in the query parameters.

    Args:
        dossier_var: The Cypher variable name for the Dossier node.

    Returns:
        A Cypher string fragment starting with "\\n      AND …"
    """
    return (
        f"\n      AND ("
        f"\n        EXISTS {{ MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}}) }}"
        f"\n        OR EXISTS {{"
        f"\n          MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)"
        f"\n               -[:`GRANT:READ`]->({dossier_var})"
        f"\n        }}"
        f"\n        OR ("
        f"\n          NOT coalesce({dossier_var}.isProtected, false)"
        f"\n          AND EXISTS {{"
        f"\n            MATCH (u:User:SystemAcl {{id: $effectiveUserId}})-[:MEMBER_OF]->(g:Group:SystemAcl)"
        f"\n                 -[:`GRANT:READ`]->(anc:Dossier)"
        f"\n                 -[:CONTAINS*1..{ACL_MAX_INHERITANCE_DEPTH}]->({dossier_var})"
        f"\n            WHERE NOT coalesce(anc.isProtected, false)"
        f"\n          }}"
        f"\n        )"
        f"\n      )"
    )


def acl_dossier_permission_check_query(permission: str) -> str:
    """Return the complete Cypher query that checks whether $effective_user_id holds
    `permission` on the Dossier identified by $dossier_id.

    Traverses the full ACL chain:
    - Tenant-admin bypass via 'tenant-admin' Capability
    - Direct GRANT:<permission> on the dossier
    - Inherited GRANT:<permission> from ancestor Dossier (stopping at isProtected)

    Parameters (must be passed by the caller):
        $effective_user_id   — Neo4j User.id of the requesting user
        $dossier_id — id property of the target Dossier node

    Args:
        permission: The permission to check, e.g. 'READ', 'ADD', 'DELETE'.

    Returns:
        A Cypher query string that yields a single row with key 'has_access' (bool).
    """
    grant_rel = f"GRANT:{permission}"
    return f"""
        MATCH (u:User:SystemAcl {{id: $effective_user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl)
        WITH collect(g.id) AS groupIds
        WITH groupIds,
             EXISTS {{
                 MATCH (tg:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}})
                 WHERE tg.id IN groupIds
             }} AS isTenantAdmin
        MATCH (d:Dossier {{id: $dossier_id}})
        WITH d, groupIds, isTenantAdmin,
             EXISTS {{
                 MATCH (gd:Group:SystemAcl)-[:`{grant_rel}`]->(d)
                 WHERE gd.id IN groupIds
             }} AS directGrant
        OPTIONAL MATCH aclPath = (d)<-[:CONTAINS*1..{ACL_MAX_INHERITANCE_DEPTH}]-(ancestor:Dossier)
        WHERE NOT coalesce(d.isProtected, false)
        WITH d, groupIds, isTenantAdmin, directGrant,
             ancestor, length(aclPath) AS aclDepth,
             coalesce(ancestor.isProtected, false) AS ancestorProtected
        ORDER BY aclDepth ASC
        WITH isTenantAdmin, directGrant, groupIds,
             collect({{id: ancestor.id, protected: ancestorProtected}}) AS ancestorChain
        WITH isTenantAdmin, directGrant, groupIds,
             reduce(s = {{ids: [], stop: false}}, x IN ancestorChain |
                 CASE WHEN s.stop THEN s
                      ELSE {{ids: s.ids + x.id, stop: x.protected}}
                 END
             ) AS aclScope
        RETURN isTenantAdmin
            OR directGrant
            OR EXISTS {{
                   MATCH (ga:Group:SystemAcl)-[:`{grant_rel}`]->(anc:Dossier)
                   WHERE ga.id IN groupIds AND anc.id IN aclScope.ids
               }} AS has_access
    """


def acl_document_permission_check_query(permission: str) -> str:
    """Return the complete Cypher query that checks whether $effective_user_id holds
    `permission` on the Document identified by $document_id.

    Traverses the full ACL chain:
    - Soft-delete exclusion: document.isDeleted = false
    - Tenant-admin bypass via 'tenant-admin' Capability
    - Direct GRANT:<permission> on the document
    - Inherited GRANT:<permission> from ancestor Dossier (stopping at isProtected)

    Parameters (must be passed by the caller):
        $effective_user_id    — Neo4j User.id of the requesting user
        $document_id — id property of the target Document node

    Args:
        permission: The permission to check, e.g. 'READ', 'EDIT', 'DELETE', 'ADD'.

    Returns:
        A Cypher query string that yields a single row with key 'has_access' (bool).
    """
    grant_rel = f"GRANT:{permission}"
    return f"""
        MATCH (d:Document {{id: $document_id}})
        WHERE d.isDeleted = false
        MATCH (u:User:SystemAcl {{id: $effective_user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl)
        WITH d, collect(g.id) AS groupIds
        WITH d, groupIds,
             EXISTS {{
                 MATCH (tg:Group:SystemAcl)-[:HAS_ROLE]->(:UserRole:SystemAcl)-[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {{id: 'tenant-admin'}})
                 WHERE tg.id IN groupIds
             }} AS isTenantAdmin
        WITH d, groupIds, isTenantAdmin,
             EXISTS {{
                 MATCH (gd:Group:SystemAcl)-[:`{grant_rel}`]->(d)
                 WHERE gd.id IN groupIds
             }} AS directGrant
        OPTIONAL MATCH aclPath = (d)<-[:CONTAINS*1..{ACL_MAX_INHERITANCE_DEPTH}]-(ancestor:Dossier)
        WHERE NOT coalesce(d.isProtected, false)
        WITH d, groupIds, isTenantAdmin, directGrant,
             ancestor, length(aclPath) AS aclDepth,
             coalesce(ancestor.isProtected, false) AS ancestorProtected
        ORDER BY aclDepth ASC
        WITH isTenantAdmin, directGrant, groupIds,
             collect({{id: ancestor.id, protected: ancestorProtected}}) AS ancestorChain
        WITH isTenantAdmin, directGrant, groupIds,
             reduce(s = {{ids: [], stop: false}}, x IN ancestorChain |
                 CASE WHEN s.stop THEN s
                      ELSE {{ids: s.ids + x.id, stop: x.protected}}
                 END
             ) AS aclScope
        RETURN isTenantAdmin
            OR directGrant
            OR EXISTS {{
                   MATCH (ga:Group:SystemAcl)-[:`{grant_rel}`]->(anc:Dossier)
                   WHERE ga.id IN groupIds AND anc.id IN aclScope.ids
               }} AS has_access
    """


# =============================================================================
# ACL Permission Creation Functions
# =============================================================================
# Standalone functions that produce Cypher queries for creating ACL relationships
# (OWNS, GRANT:*) when creating new resources.
# =============================================================================


def acl_create_owns_query(group_id: str, resource_id: str, resource_label: str = "Dossier") -> str:
    """Return the Cypher query to create an OWNS relationship from a group to a resource.

    Args:
        group_id: The Neo4j Group.id of the owner group
        resource_id: The id property of the target resource node
        resource_label: The node label (Dossier, Document, Ontology)

    Returns:
        A Cypher query string that creates the OWNS relationship
    """
    return f"""
        MATCH (g:Group:SystemAcl {{id: $group_id}}), (r:{resource_label} {{id: $resource_id}})
        MERGE (g)-[:OWNS]->(r)
        RETURN count(*) AS count
    """


def acl_public_read_grant_query(group_id: str, resource_id: str, resource_label: str = "Document") -> str:
    """Return the Cypher that publishes one resource to the public read audience (T32 D2).

    Per-document publication shape: exactly one GRANT:READ edge from the public audience
    group to the resource — never OWNS, never UPDATE/ADD/DELETE/SHARING/OWNERSHIP.
    Idempotent: MERGE on the relationship, timestamp properties only ON CREATE.

    Args:
        group_id: The public audience group id (C# Constants.PublicGroupId, 'public-group')
        resource_id: The id property of the target resource node
        resource_label: The node label (Document, Dossier, Ontology)

    Returns:
        A Cypher query string binding $group_id and $resource_id
    """
    return f"""
        MATCH (pg:Group:SystemAcl {{id: $group_id}}), (r:{resource_label} {{id: $resource_id}})
        MERGE (pg)-[g:`GRANT:READ`]->(r)
        ON CREATE SET g.grantedAt = datetime(), g.grantedBy = 'public-ingestion'
        RETURN count(g) AS count
    """


def acl_documented_entities_read_query(entity_id: str, doc_var: str = "d") -> dict:
    """Return an ACL-guarded query listing the documents that document a given entity (T27 S9/R6).

    The ``(:Entity)-[:IS_DOCUMENTED]->(:Document)`` edge reveals that a document exists and
    links it to a known entity even when the reader cannot read that document. Every read of
    the edge therefore passes through the same READ predicate as document listings
    (``acl_document_where_clause``) -- the isDeleted filter and the grant check arrive
    together and no hand-rolled visibility rule exists here to drift.

    Readers without a grant receive no identifiers: pair with
    ``acl_documented_entities_count_query`` to show an opaque "referenced by N documents"
    count instead. The generated Cypher binds the caller's group ids to $group_ids and the
    entity under expansion to $entity_id.

    Parameters (must be passed by the caller):
        $group_ids -- the effective user's group ids, as list[str]
        $entity_id -- the id property of the entity to expand, as str

    Args:
        entity_id: Bound to $entity_id; passed through for caller convenience.
        doc_var: Cypher variable name the readable Document nodes are bound to.

    Returns:
        Dict with a 'query' Cypher-string key and 'parameters' mapping key containing the
        bound id, e.g. {"query": <cypher>, "parameters": {"entity_id": "..."}}.
    """
    query = f"""
        MATCH (e:Entity {{id: $entity_id}})-[:IS_DOCUMENTED]->({doc_var}:Document)
        WHERE coalesce({doc_var}.isDeleted, false) = false
        {acl_document_where_clause(doc_var)}
        RETURN {doc_var}.id AS document_id,
               {doc_var}.name AS document_name
        ORDER BY document_name
    """
    return {"query": query, "parameters": {"entity_id": entity_id}}


def acl_documented_entities_count_query(entity_id: str, doc_var: str = "d") -> dict:
    """Return the opaque count twin of ``acl_documented_entities_read_query`` (T27 S9/R6).

    For renderers facing a zero-row gated result: report "this entity is referenced by N
    documents" from a count statement that carries no ACL predicate and returns no
    identifier -- the S9-sanctioned fallback, where a reader who cannot read a document
    learns a number, never a document id or title. Apply this only when the entity itself
    is already visible to the caller, else the count leaks entity existence.

    Parameters (must be passed by the caller):
        $entity_id -- the id property of the entity to count references for, as str

    Args:
        entity_id: Bound to $entity_id; passed through for caller convenience.
        doc_var: Cypher variable name the Document nodes are bound to (must match the
                 caller's outer MATCH; the count is ACL-free by design).

    Returns:
        Dict with a 'query' Cypher-string key and 'parameters' mapping key containing the
        bound id, e.g. {"query": <cypher>, "parameters": {"entity_id": "..."}}.
    """
    query = f"""
        MATCH (e:Entity {{id: $entity_id}})-[:IS_DOCUMENTED]->({doc_var}:Document)
        WHERE coalesce({doc_var}.isDeleted, false) = false
        RETURN count({doc_var}) AS hidden_references
    """
    return {"query": query, "parameters": {"entity_id": entity_id}}


def acl_create_grant_permissions_query(group_id: str, resource_id: str, permissions: list, resource_label: str = "Dossier") -> str:
    """Return the Cypher query to grant multiple permissions to a group on a resource.

    Uses apoc.merge.relationship to create GRANT:* relationships with properties.

    Args:
        group_id: The Neo4j Group.id of the group receiving permissions
        resource_id: The id property of the target resource node
        permissions: List of permission strings (e.g., ["READ", "SHARING", "DELETE", "ADD", "UPDATE", "OWNERSHIP"])
        resource_label: The node label (Dossier, Document, Ontology)

    Returns:
        A Cypher query string that grants all specified permissions
    """
    return f"""
        MATCH (g:Group:SystemAcl {{id: $group_id}}), (r:{resource_label} {{id: $resource_id}})
        UNWIND $permissions AS perm
        CALL apoc.merge.relationship(g, 'GRANT:' + perm, {{}},
            {{grantedAt: $grantedAt, grantedBy: $grantedBy}}, r, {{}})
        YIELD rel
        RETURN count(rel) AS count
    """


def acl_create_grant_permissions_to_user_query(user_id: str, resource_id: str, permissions: list, resource_label: str = "Dossier") -> str:
    """Return the Cypher query to grant multiple permissions to a user's group on a resource.

    This is a convenience wrapper that:
    1. Resolves the user's group (isUserGroup=true) from user_id
    2. Calls acl_create_grant_permissions_query with the resolved group_id

    Args:
        user_id: The Neo4j User.id of the user
        resource_id: The id property of the target resource node
        permissions: List of permission strings
        resource_label: The node label (Dossier, Document, Ontology)

    Returns:
        A Cypher query string that grants all specified permissions to the user's group
    """
    return f"""
        MATCH (u:User:SystemAcl {{id: $user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl {{isUserGroup: true}})
        WITH g
        MATCH (r:{resource_label} {{id: $resource_id}})
        UNWIND $permissions AS perm
        CALL apoc.merge.relationship(g, 'GRANT:' + perm, {{}},
            {{grantedAt: $grantedAt, grantedBy: $grantedBy}}, r, {{}})
        YIELD rel
        RETURN count(rel) AS count
    """


def acl_create_owns_and_grant_permissions_query(user_id: str, resource_id: str, permissions: list, resource_label: str = "Dossier") -> str:
    """Return a combined Cypher query that creates OWNS and grants all permissions to a user's group.

    This is the most common pattern for resource creation:
    1. Create OWNS relationship from user's group to the new resource
    2. Grant all specified permissions to the user's group
    3. Optionally grant all permissions to the global user (system/admin access)

    Args:
        user_id: The Neo4j User.id of the creator
        resource_id: The id property of the new resource
        permissions: List of permission strings
        resource_label: The node label (Dossier, Document, Ontology)

    Returns:
        A list of Cypher query strings and their parameters
    """
    return [
        {
            "query": f"""
                MATCH (u:User:SystemAcl {{id: $user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl {{isUserGroup: true}})
                MERGE (g)-[:OWNS]->(r:{resource_label} {{id: $resource_id}})
                RETURN count(*) AS count
            """,
            "params": {"user_id": user_id, "resource_id": resource_id}
        },
        {
            "query": f"""
                MATCH (u:User:SystemAcl {{id: $user_id}})-[:MEMBER_OF]->(g:Group:SystemAcl {{isUserGroup: true}})
                MATCH (r:{resource_label} {{id: $resource_id}})
                UNWIND $permissions AS perm
                CALL apoc.merge.relationship(g, 'GRANT:' + perm, {{}},
                    {{grantedAt: $grantedAt, grantedBy: $grantedBy}}, r, {{}})
                YIELD rel
                RETURN count(rel) AS count
            """,
            "params": {"user_id": user_id, "resource_id": resource_id, "permissions": permissions}
        }
    ]


log_end_file(__file__)
