from typing import Protocol, TypeVar, Any

T = TypeVar("T", bound=type[Any], contravariant=True)


class InstanceOf(Protocol[T]):
    __class__: T  # type: ignore


x: InstanceOf[type] = int
