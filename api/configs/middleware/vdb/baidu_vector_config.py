from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class BaiduVectorDBConfig(BaseSettings):
    """
    Configuration settings for Baidu Vector Database
    """

    BAIDU_VECTOR_DB_ACCOUNT: Optional[str] = Field(
        description="Account for authenticating with the Baidu Vector Database service",
        default="root",
    )

    BAIDU_VECTOR_DB_API_KEY: Optional[str] = Field(
        description="API key for authenticating with the Baidu Vector Database service",
        default=None,
    )

    BAIDU_VECTOR_DB_ENDPOINT: Optional[str] = Field(
        description="Endpoint of the Tencent Vector Database service (e.g., 'http://10.1.1.1:5287')",
        default=None,
    )

    BAIDU_VECTOR_DB_DATABASE: Optional[str] = Field(
        description="Name of the specific Baidu Vector Database to connect to",
        default=None,
    )

    BAIDU_VECTOR_DB_INSTANCE_ID: Optional[str] = Field(
        default=None,
        description="The unique identifier of the Baidu Vector Database instance you want to connect to.",
    )
