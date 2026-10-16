from src.interfaces.user_interface import UserInterface
from src.services.user_service import get_user_service
from src.config.constants import GLOBAL_USER_ID

def init_unit_test():
    user_service: UserInterface = get_user_service()
    user_service.set_user_uuid(GLOBAL_USER_ID)
    user_service.set_user_id(GLOBAL_USER_ID)
