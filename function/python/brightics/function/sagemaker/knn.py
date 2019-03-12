import io
import boto3
import sagemaker
from sagemaker import KNN

from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.validation import raise_runtime_error
from brightics.function.utils import _model_dict
from brightics.function.sagemaker.utils import get_logs
from brightics.function.sagemaker.utils import output_data_from_s3
from brightics.function.sagemaker.utils import kwargs_from_string
from brightics.function.sagemaker.utils import record_set
from brightics.function.sagemaker.utils import get_model_data_path


def knn_train(table,
              feature_cols,
              label_col,
              k,
              predictor_type,
              connection,
              role,
              region_name,
#             dimension_reduction_target = None,
#             dimension_reduction_type = None,
#             faiss_index_ivf_nlists = None,
#             faiss_index_pq_m = None,
#             index_metric = None,
#             index_type = None,
              sample_size=None,
              kwargstr=None,
              mini_batch_size=5000,
              wait=True,
              job_name=None,
              train_instance_count=1,
              train_instance_type='ml.m5.large',
              train_volume_size=30,
              train_max_run=24 * 60 * 60
              ):

    kwargs = {}
    if kwargstr is not None:
        kwargs = kwargs_from_string(kwargstr)

    features_arr = table[feature_cols].values.astype('float32')
    label_arr = table[label_col].values.astype('float32')

    num_records = features_arr.shape[0]
    mini_batch_size = min(mini_batch_size, num_records)
    if sample_size is None:
        sample_size = num_records

    aws_access_key_id = connection['accessKeyId']
    aws_secret_access_key = connection['secretAccessKey']

    boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)

    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sagemaker_client = sagemaker_session.sagemaker_client

    sagemaker_knn = KNN(role=role,
                        k=k,
                        train_instance_count=train_instance_count,
                        train_instance_type=train_instance_type,
                        train_volume_size=train_volume_size,
                        train_max_run=train_max_run,
                        sample_size=sample_size,
                        predictor_type=predictor_type,
#                         dimension_reduction_type=dimension_reduction_type,
#                         dimension_reduction_target=dimension_reduction_target,
#                         index_type=index_type,
#                         index_metric=index_metric,
#                         faiss_index_ivf_nlists=faiss_index_ivf_nlists,
#                         faiss_index_pq_m=faiss_index_pq_m,
                        sagemaker_session=sagemaker_session,
                        **kwargs)

    try:
        sagemaker_knn.fit(sagemaker_knn.record_set(features_arr, label_arr),
                        mini_batch_size=mini_batch_size,
                        wait=wait, job_name=job_name)
    finally:
        if sagemaker_knn.latest_training_job is not None:
            sagemaker_client = sagemaker_session.sagemaker_client

            job_name = sagemaker_knn.latest_training_job.name
            model_data = get_model_data_path(sagemaker_knn)
            url = '''<https://{}.console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}>'''.format(
                region_name, region_name, job_name)
            description = sagemaker_client.describe_training_job(TrainingJobName=job_name)

            if wait is False:
                message = ('This function terminated with training job status: {status}.'
                ' Please visit the following link to get more information on the training job.\n\n'
                '- {url}'.format(status=description['TrainingJobStatus'], url=url))

            elif description['TrainingJobStatus'] == 'Completed':
                message = ('This function terminated with training job status: {status}.'
                ' Please visit the following link to get more information on the training job.\n\n'
                '- {url}'.format(status=description['TrainingJobStatus'], url=url))

            elif description['TrainingJobStatus'] == 'Failed':
                failure_reason = description['FailureReason']
                message = ('This function failed with the following error: {failure_reason}.'
                ' Please visit the following link to get more information on the training job.\n\n'
                '- {url}'.format(failure_reason=failure_reason, url=url))

            else:
                sagemaker_client.stop_training_job(TrainingJobName=job_name)
                message = ('This function terminated unexpectedly.'
                ' Please visit AWS management console to make sure that'
                ' the training job is properly stopped.\n\n- {url}'.format(url=url))

            logs = get_logs(sagemaker_session, job_name)

            rb = BrtcReprBuilder()
            rb.addMD(strip_margin('''
            | ## Sagemaker k-nearest neighbors (k-NN) Train Result
            | {message}
            |
            | ### Logs
            | ```
            | {logs}
            | ```'''.format(message=message, logs=logs)))

            model = _model_dict('sagemaker.knn_train')
            model['feature_cols'] = feature_cols
            model['label_col'] = label_col
            model['region_name'] = region_name
            model['role'] = role
            model['model_data'] = model_data
            model['description'] = description
            model['_repr_brtc_'] = rb.get()

            return {'model':model}


def knn_predict(
    table, model,
    connection,
    prediction_col='prediction',
    instance_count=1, instance_type='ml.m5.large'):

    if model['_context'] == 'python' and model['_type'] == 'sagemaker.knn_train':
        aws_access_key_id = connection['accessKeyId']
        aws_secret_access_key = connection['secretAccessKey']

        region_name = model['region_name']
        model_path = model['model_data']
        role = model['role']

        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)

        feature_cols = model['feature_cols']

        record = record_set(sagemaker_session=sagemaker_session,
            features=table[feature_cols].values.astype('float32'),
            labels=None, channel="train", num_shards=instance_count, prefix='brightics')

        knn_model = sagemaker.KNNModel(model_data=model_path,
                                       role=role,
                                       sagemaker_session=sagemaker_session)
        transformer = knn_model.transformer(instance_count=instance_count,
                                            instance_type=instance_type,
                                            accept='application/json')

        try:
            transformer.transform(record.s3_data,
                                  data_type='ManifestFile',
                                  content_type='application/x-recordio-protobuf',
                                  split_type='RecordIO')
            transformer.wait()

            outputs = output_data_from_s3(boto_session, transformer.output_path)
            prediction = []
            for _, output in sorted(outputs.items()):
                prediction.extend([obj['predicted_label'] for obj in output['predictions']])

            out_table = table.copy()
            out_table[prediction_col] = prediction

        finally:
            if transformer.latest_transform_job is not None:
                sagemaker_client = sagemaker_session.sagemaker_client

                job_name = transformer.latest_transform_job.name
                description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
                if description['TransformJobStatus'] != 'Completed' and \
                    description['TransformJobStatus'] != 'Failed':
                    sagemaker_client.stop_transform_job(TransformJobName=job_name)

    else:
        raise_runtime_error("Unsupported model")

    return {'out_table': out_table}
