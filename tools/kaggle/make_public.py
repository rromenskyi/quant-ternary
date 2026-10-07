"""Make every Kaggle model page in cards.json public (models are created
private). Run only on the user's go-ahead.

    KAGGLE_API_TOKEN=... python make_public.py cards.json
"""
import json
import sys

import kagglehub
from google.protobuf.field_mask_pb2 import FieldMask
from kagglehub.clients import build_kaggle_client
from kagglesdk.models.types.model_api_service import ApiGetModelRequest, ApiUpdateModelRequest

user = kagglehub.whoami()["username"]
with build_kaggle_client() as c:
    api = c.models.model_api_client
    for slug in json.load(open(sys.argv[1])):
        r = ApiUpdateModelRequest()
        r.owner_slug, r.model_slug, r.is_private = user, slug, False
        r.update_mask = FieldMask(paths=["is_private"])
        try:
            api.update_model(r)
        except Exception as e:
            if "Expecting value" not in str(e):
                print(f"{slug}: FAILED {str(e)[:120]}", flush=True)
                continue
        g = ApiGetModelRequest()
        g.owner_slug, g.model_slug = user, slug
        print(f"{slug}: is_private={api.get_model(g).is_private}  https://www.kaggle.com/models/{user}/{slug}", flush=True)
