__version__ = '0.2.2'


from .bucket import Bucket
from .config import BucketConfig, CuratorConfig, DeduperConfig, BucketWorkerManagerConfig
from .curator import Curator
from .deduper import Deduper
from .workers.bucket_worker import BucketWorkerManager

__all__ = [
    "Bucket", "BucketConfig",
    "BucketWorkerManager", "BucketWorkerManagerConfig",
    "Curator", "CuratorConfig", "Deduper", "DeduperConfig"
]
