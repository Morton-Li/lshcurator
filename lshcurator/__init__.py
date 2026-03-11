__version__ = '0.1.1'


from .bucket import Bucket
from .config import BucketConfig, CuratorConfig, DeduperConfig
from .curator import Curator
from .deduper import Deduper

__all__ = ["Bucket", "BucketConfig", "Curator", "CuratorConfig", "Deduper", "DeduperConfig"]
