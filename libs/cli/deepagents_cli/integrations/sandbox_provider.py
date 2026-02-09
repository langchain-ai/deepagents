from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol


class SandboxError(Exception):
    @property
    def original_exc(self) -> BaseException | None:
        return self.__cause__


class SandboxNotFoundError(SandboxError):
    pass


MetadataT_co = TypeVar("MetadataT_co", covariant=True)


class SandboxInfo(dict[str, Any], Generic[MetadataT_co]):
    pass


class SandboxListResponse(dict[str, Any], Generic[MetadataT_co]):
    pass


class SandboxProvider(ABC, Generic[MetadataT_co]):
    @abstractmethod
    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT_co]:
        raise NotImplementedError

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    async def alist(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT_co]:
        return await asyncio.to_thread(self.list, cursor=cursor, **kwargs)

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        return await asyncio.to_thread(
            self.get_or_create, sandbox_id=sandbox_id, **kwargs
        )

    async def adelete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)
