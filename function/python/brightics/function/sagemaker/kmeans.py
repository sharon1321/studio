import boto3
import sagemaker

from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.validation import raise_runtime_error
from brightics.function.sagemaker.utils import get_logs
from brightics.function.sagemaker.utils import record_set
from brightics.function.sagemaker.utils import output_data_from_s3
from brightics.function.sagemaker.utils import kwargs_from_string
from brightics.function.utils import _model_dict


def kmeans_train(table,
    input_cols, k,
    connection, role, region_name,
    kwargstr=None,
    mini_batch_size=None, wait=True, job_name=None,
    train_instance_count=1, train_instance_type='ml.m4.xlarge',
    train_volume_size=30, train_max_run=24 * 60 * 60,
    data_location=None, output_path=None):

    kwargs = {}
    if kwargstr is not None:
        kwargs = kwargs_from_string(kwargstr)

    aws_access_key_id = connection['accessKeyId']
    aws_secret_access_key = connection['secretAccessKey']

    boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)

    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    sagemaker_kmeans = sagemaker.KMeans(
        role=role, sagemaker_session=sagemaker_session,
        train_instance_count=train_instance_count,
        train_instance_type=train_instance_type,
        train_volume_size=train_volume_size,
        train_max_run=train_max_run,
        data_location=data_location,
        output_path=output_path,

        k=k,
        **kwargs)

    data_arr = table[input_cols].values.astype('float32')

    try:
        sagemaker_kmeans.fit(sagemaker_kmeans.record_set(data_arr),
            mini_batch_size=mini_batch_size, wait=wait, job_name=job_name)

    finally:
        if sagemaker_kmeans.latest_training_job is not None:
            sagemaker_client = sagemaker_session.sagemaker_client

            job_name = sagemaker_kmeans.latest_training_job.name
            model_data = sagemaker_kmeans.model_data
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
            | ## Sagemaker K-Means Train Result
            | {message}
            |
            | ### Logs
            | ```
            | {logs}
            | ```'''.format(message=message, logs=logs)))

            model = _model_dict('sagemaker.kmeans_train')
            model['input_cols'] = input_cols
            model['k'] = k
            model['region_name'] = region_name
            model['role'] = role
            model['model_data'] = model_data
            model['description'] = description
            model['_repr_brtc_'] = rb.get()

            return {'model':model}


def kmeans_predict(
    table, model,
    connection,
    prediction_col='prediction',
    distance_to_cluster_col='distance_to_cluster',
    instance_count=1, instance_type='ml.m5.large'):

    if model['_context'] == 'python' and model['_type'] == 'sagemaker.kmeans_train':
        aws_access_key_id = connection['accessKeyId']
        aws_secret_access_key = connection['secretAccessKey']

        region_name = model['region_name']
        model_data = model['model_data']
        role = model['role']

        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)

        input_cols = model['input_cols']

        record = record_set(sagemaker_session=sagemaker_session,
            features=table[input_cols].values.astype('float32'),
            labels=None, num_shards=instance_count, prefix='brightics')

        kmeans_model = sagemaker.KMeansModel(model_data=model_data,
                                             role=role,
                                             sagemaker_session=sagemaker_session)
        transformer = kmeans_model.transformer(instance_count=instance_count,
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
            distance = []
            for _, output in sorted(outputs.items()):
                prediction.extend([obj['closest_cluster'] for obj in output['predictions']])
                distance.extend([obj['distance_to_cluster'] for obj in output['predictions']])

            out_table = table.copy()
            out_table[prediction_col] = prediction
            out_table[distance_to_cluster_col] = distance

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

    return {'out_table':out_table}
