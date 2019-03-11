from contextlib import redirect_stdout
from io import StringIO
import logging
import json

from sagemaker.utils import sagemaker_timestamp
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.fw_utils import parse_s3_url
from sagemaker.amazon.amazon_estimator import upload_numpy_to_s3_shards



def get_logs(sagemaker_session, job_name):
    with StringIO() as buf, redirect_stdout(buf):
        sagemaker_session.logs_for_job(job_name)
        return buf.getvalue()


def kwargs_from_string(kwargstrs):
    return eval('dict({})'.format(kwargstrs))

# from sagemaker.amazon.amazon_estimator.record_set
def record_set(sagemaker_session, features, labels=None, channel="train", num_shards=1, data_location=None, prefix='brightics'):
    logger = logging.getLogger(__name__)

    if data_location is None:
        data_location = "s3://{}/sagemaker-record-sets/".format(sagemaker_session.default_bucket())

    s3 = sagemaker_session.boto_session.resource('s3')
    bucket, key_prefix = parse_s3_url(data_location)
    key_prefix = key_prefix + '{}-{}/'.format(prefix, sagemaker_timestamp())
    key_prefix = key_prefix.lstrip('/')

    logger.debug('Uploading to bucket {} and key_prefix {}'.format(bucket, key_prefix))
    manifest_s3_file = upload_numpy_to_s3_shards(num_shards, s3, bucket, key_prefix, features, labels)
    logger.debug("Created manifest file {}".format(manifest_s3_file))

    return RecordSet(manifest_s3_file, num_records=features.shape[0], feature_dim=features.shape[1], channel=channel)


def output_data_from_s3(session, s3_data_path, accept='application/json', assemble_with = None):
    s3 = session.resource('s3')

    if accept == 'application/json':
        bucket, prefix = parse_s3_url(s3_data_path)
        object_summaries = s3.Bucket(bucket).objects.filter(Prefix=prefix)
        result = {}
        for shard_index, obj_summary in enumerate(object_summaries): #todo multiprocess?
            obj = s3.meta.client.get_object(Bucket=obj_summary.bucket_name, Key=obj_summary.key)
            result[shard_index] = json.loads(obj['Body'].read())

    else:
        raise NotImplementedError('not implemented yet.')

    return result
    