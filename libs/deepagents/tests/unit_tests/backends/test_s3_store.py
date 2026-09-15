import io
from typing import Any

from deepagents.backends import S3Store


class NoSuchKeyError(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def get_object(self, **kwargs: Any):
        try:
            body = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        except KeyError:
            raise NoSuchKeyError from None
        return {"Body": io.BytesIO(body)}

    def put_object(self, **kwargs: Any):
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        return {}

    def delete_object(self, **kwargs: Any):
        self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)
        return {}

    def list_objects_v2(self, **kwargs: Any):
        keys = sorted(key for bucket, key in self.objects if bucket == kwargs["Bucket"] and key.startswith(kwargs["Prefix"]))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}


def test_s3_store_round_trip_and_delete() -> None:
    client = FakeS3Client()
    store = S3Store(client=client, bucket="bucket", prefix="app/store")

    store.put(("users", "alice"), "profile", {"theme": "dark"})

    item = store.get(("users", "alice"), "profile")
    assert item is not None
    assert item.value == {"theme": "dark"}
    store.delete(("users", "alice"), "profile")
    assert store.get(("users", "alice"), "profile") is None


def test_s3_store_escapes_namespace_and_key_components() -> None:
    client = FakeS3Client()
    store = S3Store(client=client, bucket="bucket", prefix="root")

    store.put(("tenant/name", "nested/name"), "../blob/name", {"value": 1})

    object_key = next(iter(client.objects))[1]
    assert object_key.startswith("root/")
    assert "../" not in object_key
    assert store.get(("tenant/name", "nested/name"), "../blob/name").value == {"value": 1}


def test_s3_store_search_filter_and_namespaces() -> None:
    store = S3Store(client=FakeS3Client(), bucket="bucket")
    store.put(("users", "alice"), "a", {"kind": "note", "meta": {"score": 2}})
    store.put(("users", "alice"), "b", {"kind": "task", "meta": {"score": 1}})
    store.put(("users", "bob"), "c", {"kind": "note", "meta": {"score": 3}})

    results = store.search(("users",), filter={"kind": "note", "meta": {"score": {"$gt": 2}}})

    assert [item.key for item in results] == ["c"]
    assert store.list_namespaces(prefix=("users", "*"), max_depth=2) == [("users", "alice"), ("users", "bob")]


async def test_s3_store_async_round_trip() -> None:
    store = S3Store(client=FakeS3Client(), bucket="bucket")

    await store.aput(("blobs",), "abc", {"base64": "YWJj"})

    item = await store.aget(("blobs",), "abc")
    assert item is not None
    assert item.value == {"base64": "YWJj"}
