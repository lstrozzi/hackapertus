"""
Thalabus LLM Provider Implementation

This provider extends HTTP request functionality for Thalabus-specific operations.
"""

import email.utils
import traceback
from typing import TYPE_CHECKING, List, Dict, Union
from datetime import datetime, timezone
from requests import Response
import requests

from src.core.llm.providers.provider_http import LlmHttpRequest
from src.core.llm import MAX_COMPLETION_TOKENS, adjust_endpoint_to_docker_format
from src.core.base.InputOutput import get_document_path, get_last_modified
from src.utils.log import log, ERROR

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent


class LlmThalabus(LlmHttpRequest):
    """Thalabus LLM provider with specialized document processing capabilities."""
    
    def __init__(self):
        super().__init__()
        self._model_provider = "thalabus"

    def _prepare_model_client(self, agent: 'Agent' = None, purpose: str = None, endpoint=None):
        """Prepare and resolve the API token for Thalabus.
        
        Handles token resolution from llmconfig, including environment variable expansion
        using the {{ENV_VAR}} format. If no valid token is found, _api_key remains None.
        """
        self._api_key = None
        if self.llmconfig and "token" in self.llmconfig and self.llmconfig["token"]:
            token = self.llmconfig["token"]
            self._api_key = self._resolve_api_key(token)
        return None

    def _get(self, agent: 'Agent', purpose: str, url: str, last_modified_str: str) -> Response:
        """Make authenticated GET request to Thalabus endpoint."""
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        llm_config = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        api_token = llm_config.get("token")
        if not api_token:
            log(ERROR, "No token found in Thalabus LLM configuration")
            return None
        
        resolved_token = self._resolve_api_key(api_token)
        if not resolved_token:
            log(ERROR, "Token resolution failed for Thalabus provider (token is null or environment variable not found)")
            return None
        
        endpoint = adjust_endpoint_to_docker_format(llm_config["endpoint"] + url)

        headers = None
        if resolved_token and resolved_token.strip():
            headers = {
                "Authorization": f"Bearer {resolved_token.strip()}"
            }

        if last_modified_str:
            try:
                last_modified_dt = datetime.strptime(last_modified_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                last_modified_http = email.utils.format_datetime(last_modified_dt)
                if last_modified_http:
                    if headers is None:
                        headers = {}
                    headers["If-Modified-Since"] = last_modified_http
            except (ValueError, TypeError) as e:
                log(ERROR, f"Error parsing last modified date {last_modified_str}: {e}")

        try:
            response = requests.get(endpoint, headers=headers)
            return response
        
        except requests.exceptions.RequestException as e:
            stack = traceback.format_exc()
            log(ERROR, f"ERROR: LlmThalabus: the HTTP request to the endpoint failed with exception: {e}\nStack trace:\n{stack}")
            return None

    def _post(self, agent: 'Agent', purpose: str, url: str, data: Dict, files: List[str] = None) -> Union[str, Dict]:
        """Make authenticated POST request to Thalabus endpoint."""
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        llm_config = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        api_token = llm_config["token"]
        endpoint = adjust_endpoint_to_docker_format(llm_config["endpoint"] + url)

        headers = {
            "Authorization": f"Bearer {api_token}"
        }

        file_data = {}
        if files:
            # Use a consistent field name, e.g. "file"
            for file_path in files:
                try:
                    file_name = file_path.split('/')[-1]
                    file_data["file"] = (file_name, open(file_path, 'rb'))
                except (OSError, IOError) as e:
                    log(ERROR, f"Error opening file {file_path}: {e}")
                    return None
        else:
            headers["Content-Type"] = "application/json"

        try:
            # Switch to data=... for form data upload
            response = requests.post(endpoint, json=data, headers=headers, files=file_data)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            stack = traceback.format_exc()
            log(ERROR, f"ERROR: LlmThalabus: the HTTP request to the endpoint failed with exception: {e}\nStack trace:\n{stack}")
            return None
        
        finally:
            # Close any opened files
            for _, file_handle in file_data.values():
                if hasattr(file_handle, 'close'):
                    file_handle.close()

    def clear_directories(self, agent: 'Agent' = None, purpose: str = None) -> Dict:
        """
        Clear directories on the Thalabus server.
        POST "http://<hostname>:<port>/clear"
        """
        response = self._post(agent, purpose, "clear", {})
        return response

    def get_pdf(self, pdf_filename: str, agent: 'Agent' = None, purpose: str = None) -> Response:
        """
        Get PDF file from Thalabus server.
        GET "http://<hostname>:<port>/get/pdf/file.pdf"
        
        Note: Sends last modified datetime in the If-Modified-Since header field,
        so the server can return a 304 Not Modified response if the file exists and has not been modified.
        """
        pdf_filepath = get_document_path(pdf_filename)
        last_modified = get_last_modified(pdf_filepath)

        url = f"pdf/{pdf_filename}"
        response = self._get(agent, purpose, url, last_modified)
        return response

    def upload_pdf(self, pdf_filename: str, agent: 'Agent' = None, purpose: str = None) -> Dict:
        """
        Upload PDF file to Thalabus server.
        First checks if the PDF file exists in the cache, then uploads if necessary.
        POST -F "file=@/path/to/your/file.pdf" "http://<hostname>:<port>/upload/pdf"
        """
        # First check if the PDF file exists in the cache
        exists = self.get_pdf(pdf_filename, agent, purpose)
        if exists and exists.status_code == 200:
            return {"status": "success", "file_url": pdf_filename}
        
        # Upload the PDF file
        url = "upload/pdf"
        pdf_filepath = get_document_path(pdf_filename)
        data = {}
        files = [pdf_filepath]
        response = self._post(agent, purpose, url, data, files=files)
        return response

    def process_pdf(self, pdf_filename: str, agent: 'Agent' = None, purpose: str = None) -> Dict:
        """
        Process PDF file on Thalabus server.
        POST "http://<hostname>:<port>/process/pdf/file.pdf"
        """
        url = f"process/pdf/{pdf_filename}"
        response = self._post(agent, purpose, url, {})
        return response

    def process_image(self, pdf_filename: str, image_nr: int, agent: 'Agent' = None, purpose: str = None) -> Dict:
        """
        Process specific image from PDF file on Thalabus server.
        POST "http://<hostname>:<port>/process/image/file.pdf/image_nr"
        """
        url = f"process/image/{pdf_filename}/{image_nr}"
        response = self._post(agent, purpose, url, {})
        return response

    def prompt(self, prompt: str, pdf_filename: str, image_nr: int, agent: 'Agent' = None, purpose: str = None) -> Union[str, Dict]:
        """
        Send prompt for processing with PDF and image context.
        """
        url = f"prompt?max_new_tokens={MAX_COMPLETION_TOKENS}"
        data = {
            "prompt": prompt,
            "pdf_filename": pdf_filename,
            "image_nr": image_nr
        }

        response = self._post(agent, purpose, url, data)
        return response