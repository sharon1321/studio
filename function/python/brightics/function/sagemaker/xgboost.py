import io
import copy
from time import gmtime
from time import strftime

import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.validation import raise_runtime_error
from brightics.function.sagemaker.utils import get_logs
from brightics.function.sagemaker.utils import kwargs_from_string
from brightics.function.utils import _model_dict


def to_libsvm(f, labels, values):
    f.write(bytes('\n'.join(
        ['{} {}'.format(label, ' '.join(['{}:{}'.format(i + 1, el) for i, el in enumerate(vec)])) for label, vec in
         zip(labels, values)]), 'utf-8'))
    return f


def write_to_s3(fobj, bucket, key, s3_client):
    return s3_client.Bucket(bucket).Object(key).upload_fileobj(fobj)


def upload_to_s3(labels, vectors, prefix, bucket, s3_client, num_partition):
    partition_bound = int(len(labels) / num_partition)
    for i in range(num_partition):
        f = io.BytesIO()
        to_libsvm(f, labels[i * partition_bound:(i + 1) * partition_bound],
                  vectors[i * partition_bound:(i + 1) * partition_bound])
        f.seek(0)
        key = "{}/train/input{}".format(prefix, str(i))
        write_to_s3(f, bucket, key, s3_client)


def xgboost_train(table, 
    feature_cols, label_col, num_class, 
    connection, role, region_name,
    max_depth=5, num_round=10,
    kwargstr=None,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size=30,
    max_runtime=24 * 60 * 60,
    wait=True, objective='multi:softmax'):

    kwargs = {}
    if kwargstr is not None:
        kwargs = kwargs_from_string(kwargstr)

    aws_access_key_id = connection['accessKeyId']
    aws_secret_access_key = connection['secretAccessKey']

    boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)

    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sagemaker_client = sagemaker_session.sagemaker_client

    role = sagemaker_session.expand_role(role)

    bucket = sagemaker_session.default_bucket()
    prefix = "xgboost-{}".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region_name, bucket)

    s3 = boto_session.resource('s3')
    container = get_image_uri(region_name, 'xgboost')

    common_training_params = \
    {
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": bucket_path + "/" + prefix
        },
        "ResourceConfig": {
            "InstanceCount": instance_count,
            "InstanceType": instance_type,
            "VolumeSizeInGB": volume_size
        },
        "HyperParameters": {
            "max_depth":str(max_depth),
            "objective": str(objective),
            "num_class": str(num_class),
            "num_round": str(num_round)
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": max_runtime
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": bucket_path + "/" + prefix + "/train",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "libsvm",
                "CompressionType": "None"
            }
        ]
    }

    common_training_params["HyperParameters"].update(kwargs)

    input_data = table[feature_cols].values.astype('float32')
    label_data = table[label_col].values.astype('int')

    upload_to_s3(label_data, input_data, prefix, bucket, s3, instance_count)
    job_name = 'sagemaker-xgboost-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    try:
        if instance_count == 1:

            single_machine_job_params = copy.deepcopy(common_training_params)
            single_machine_job_params['TrainingJobName'] = job_name
            single_machine_job_params['OutputDataConfig']['S3OutputPath'] = \
                bucket_path + "/" + prefix + "/xgboost-single"
            single_machine_job_params['ResourceConfig']['InstanceCount'] = 1

            sagemaker_client.create_training_job(**single_machine_job_params)
            if wait is True:
                sagemaker_client.get_waiter('training_job_completed_or_stopped').wait(
                    TrainingJobName=job_name)
        else:
            distributed_job_params = copy.deepcopy(common_training_params)
            distributed_job_params['TrainingJobName'] = job_name
            distributed_job_params['OutputDataConfig']['S3OutputPath'] = \
                bucket_path + "/" + prefix + "/xgboost-distributed"
            distributed_job_params['ResourceConfig']['InstanceCount'] = instance_count

            sagemaker_client.create_training_job(**distributed_job_params)
            distributed_job_params['InputDataConfig'][0]['DataSource']['S3DataSource']['S3DataDistributionType'] = 'ShardedByS3Key'
            if wait is True:
                sagemaker_client.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)

        status = sagemaker_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']

    finally:
        url = '''<https://{}.console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}>'''.format(
                region_name, region_name, job_name)

        if wait is False:
            description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            message = ('This function terminated with training job status: {status}.'
            ' Please visit the following link to get more information on the training job.\n\n'
            '- {url}'.format(status=description['TrainingJobStatus'], url=url))
        elif status == "Completed":
            description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            model_data = description['ModelArtifacts']['S3ModelArtifacts']
            message = ('This function terminated with training job status: {status}.'
            ' Please visit the following link to get more information on the training job.\n\n'
            '- {url}'.format(status=description['TrainingJobStatus'], url=url))
        elif status == 'Failed':
            description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            model_data = description['ModelArtifacts']['S3ModelArtifacts']
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
        | ## Sagemaker XGBoost Train Result
        | {message}
        |
        | ### Logs
        | ```
        | {logs}
        | ```'''.format(message=message, logs=logs)))

        model = _model_dict('sagemaker.xgboost_train')
        model['feature_cols'] = feature_cols
        model['label_col'] = label_col
        model['region_name'] = region_name
        model['role'] = role
        model['model_data'] = model_data
        model['description'] = description
        model['_repr_brtc_'] = rb.get()

        return {'model':model}


def to_libsvm_predict(f, values):
    f.write(bytes('\n'.join(
        ['{}'.format(' '.join(['{}:{}'.format(i + 1, el) for i, el in enumerate(vec)])) for vec in
          values]), 'utf-8'))
    return f


def upload_prediction_input(np_arr, s3, bucket, format='%.10f'):

    key = "{}-prediction-input-{}".format('xgboost', strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    path = "s3://{}/brightics-studio-prediction-input/{}".format(bucket, key)
    input_key = "{}/{}".format("brightics-studio-prediction-input", key)

    f = io.BytesIO()
    to_libsvm_predict(f, np_arr)
    f.seek(0)

    s3.Bucket(bucket).Object(input_key).upload_fileobj(f)

    return path, key


def xgboost_predict(
    table, model,
    connection,
    prediction_col='prediction',
    instance_count=1, instance_type='ml.m5.large'):

    if model['_context'] == 'python' and model['_type'] == 'sagemaker.xgboost_train':
        region_name = model['region_name']
        model_data = model['model_data']
        role = model['role']
        feature_cols = model['feature_cols']

        container = get_image_uri(region_name, 'xgboost')
        aws_access_key_id = connection['accessKeyId']
        aws_secret_access_key = connection['secretAccessKey']

        boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                     aws_secret_access_key=aws_secret_access_key,
                                     region_name=region_name)

        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        xgboost_model = sagemaker.model.Model(model_data=model_data, role=role, image=container,
             sagemaker_session=sagemaker_session)

        bucket = sagemaker_session.default_bucket()
        s3_client = boto_session.resource('s3')

        data_np_arr = table[feature_cols].values.astype('float32')
        input_path, input_key = upload_prediction_input(data_np_arr, s3_client, bucket)

        output_prefix = 'brightics-studio-prediction-output'
        output_data_key = '{}/{}.out'.format(output_prefix, input_key)
        output_path = 's3://{}/{}'.format(bucket, output_prefix)

        batch_transformer = xgboost_model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_path)

        try:
            batch_transformer.transform(input_path, content_type='text/libsvm', split_type='Line')
            batch_transformer.wait()

            f_output = io.BytesIO()
            s3_client.Bucket(bucket).Object(output_data_key).download_fileobj(f_output)
            f_output.seek(0)
            wrapper = io.TextIOWrapper(f_output, encoding='utf-8')
            result_str = wrapper.read()
            result_list = result_str.split()

            out_table = table.copy()
            out_table[prediction_col] = result_list

        finally:
            if batch_transformer.latest_transform_job is not None:
                sagemaker_client = sagemaker_session.sagemaker_client
                job_name = batch_transformer.latest_transform_job.name
                description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
                if description['TransformJobStatus'] != 'Completed' and \
                    description['TransformJobStatus'] != 'Failed':
                    sagemaker_client.stop_transform_job(TransformJobName=job_name)
    else:
        raise_runtime_error("Unknown Error")

    return {'out_table':out_table}
