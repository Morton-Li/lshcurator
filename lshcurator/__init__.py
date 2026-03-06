__version__ = '0.0.2'


from .bucket import Bucket
from .config import BucketConfig, CuratorConfig
from .curator import Curator
from .deduper import Deduper

__all__ = ["Bucket", "BucketConfig", "Curator", "CuratorConfig", "Deduper"]
