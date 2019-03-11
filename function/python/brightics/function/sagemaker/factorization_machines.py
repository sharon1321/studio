import boto3
import sagemaker

from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.validation import raise_runtime_error
from brightics.function.utils import _model_dict
from brightics.function.sagemaker.utils import get_logs
from brightics.function.sagemaker.utils import output_data_from_s3
from brightics.function.sagemaker.utils import kwargs_from_string
from brightics.function.sagemaker.utils import record_set


def factorization_machines_train(table,
    feature_cols, label_col, num_factors, predictor_type,
    connection, role, region_name,
    epochs=100, kwargstr=None,
    mini_batch_size=None, wait=True, job_name=None,
    train_instance_count=1, train_instance_type='ml.m5.large',
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

    sagemaker_fm = sagemaker.FactorizationMachines(
        role=role, sagemaker_session=sagemaker_session,
        train_instance_count=train_instance_count,
        train_instance_type=train_instance_type,
        train_volume_size=train_volume_size,
        train_max_run=train_max_run,
        data_location=data_location,
        output_path=output_path,

        num_factors=num_factors, predictor_type=predictor_type, epochs=epochs,
        **kwargs)

    features_arr = table[feature_cols].values.astype('float32')
    label_arr = table[label_col].values.astype('float32')

    try:
        sagemaker_fm.fit(sagemaker_fm.record_set(features_arr, label_arr),
                        mini_batch_size=mini_batch_size,
                        wait=wait, job_name=job_name)
    finally:
        if sagemaker_fm.latest_training_job is not None:
            sagemaker_client = sagemaker_session.sagemaker_client

            job_name = sagemaker_fm.latest_training_job.name
            model_data = sagemaker_fm.model_data
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
            | ## Sagemaker Factorization Machines Train Result
            | {message}
            |
            | ### Logs
            | ```
            | {logs}
            | ```'''.format(message=message, logs=logs)))

            model = _model_dict('sagemaker.factorization_machines_train')
            model['feature_cols'] = feature_cols
            model['label_col'] = label_col
            model['predictor_type'] = predictor_type
            model['region_name'] = region_name
            model['role'] = role
            model['model_data'] = model_data
            model['description'] = description
            model['_repr_brtc_'] = rb.get()

            return {'model':model}


def factorization_machines_predict(
    table, model,
    connection,
    prediction_col='prediction',
    instance_count=1, instance_type='ml.m5.large'):

    if model['_context'] == 'python' and model['_type'] == 'sagemaker.factorization_machines_train':
        aws_access_key_id = connection['accessKeyId']
        aws_secret_access_key = connection['secretAccessKey']

        region_name = model['region_name']
        model_data = model['model_data']
        role = model['role']
        predictor_type = model['predictor_type']

        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)

        feature_cols = model['feature_cols']

        record = record_set(sagemaker_session=sagemaker_session,
            features=table[feature_cols].values.astype('float32'),
            labels=None, channel="train", num_shards=instance_count, prefix='brightics')

        fm_model = sagemaker.FactorizationMachinesModel(model_data=model_data,
                                                        role=role,
                                                        sagemaker_session=sagemaker_session)
        transformer = fm_model.transformer(instance_count=instance_count,
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
                if predictor_type == 'regressor':
                    prediction.extend([obj['score'] for obj in output['predictions']])
                else:
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

    return {'out_table':out_table}
