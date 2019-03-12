import boto3
import sagemaker
from sagemaker import LinearLearner

from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.validation import raise_runtime_error
from brightics.function.utils import _model_dict
from brightics.function.sagemaker.utils import get_logs
from brightics.function.sagemaker.utils import output_data_from_s3
from brightics.function.sagemaker.utils import kwargs_from_string
from brightics.function.sagemaker.utils import record_set
from brightics.function.sagemaker.utils import get_model_data_path


def linear_learner_train(table,
                         feature_cols,
                         label_col,
                         predictor_type,
                         connection,
                         role,
                         region_name,
                         num_classes=None,
                         kwargstr=None,
                         mini_batch_size=1000,
                         wait=True,
                         job_name=None,
                         train_instance_count=1,
                         train_instance_type='ml.m5.large',
                         train_volume_size=30,
                         train_max_run=24 * 60 * 60,
#                         binary_classifier_model_selection_criteria='accuracy',
#                         target_recall=0.8,
#                         target_precision=0.8,
#                         positive_example_weight_mult=1.0,
#                         epochs=10,
#                         use_bias=True,
#                         num_models=None,
#                         num_calibration_samples=None,
#                         init_method='uniform',
#                         init_scale=0.07,
#                         init_sigma=0.01,
#                         init_bias=0,
#                         optimizer=None,
#                         loss=None,
#                         wd=0.0,
#                         l1=0.0,
#                         momentum=0,
#                         learning_rate=None,
#                         beta_1=0.9,
#                         beta_2=0.999,
#                         bias_lr_mult=10,
#                         bias_wd_mult=0,
#                         use_lr_scheduler=True,
#                         lr_scheduler_step=100,
#                         lr_scheduler_factor=0.99,
#                         lr_scheduler_minimum_lr=0.00001,
#                         normalize_data=True,
#                         normalize_label=None,
#                         unbias_data=None,
#                         unbias_label=None,
#                         num_point_for_scaler=10000,
#                         margin=1.0,
#                         quantile=0.5,
#                         loss_insensitivity=0.01,
#                         huber_delta=1.0,
#                         early_stopping_patience=3,
#                         early_stopping_tolerance=0.001,
#                         accuracy_top_k=None,
#                         f_beta=None,
#                         balance_multiclass_weights=None
                         ):
    kwargs = {}
    if kwargstr is not None:
        kwargs = kwargs_from_string(kwargstr)

    features_arr = table[feature_cols].values.astype('float32')
    label_arr = table[label_col].values.astype('float32')

    num_records = features_arr.shape[0]
    mini_batch_size = min(mini_batch_size, num_records)

    if predictor_type != 'multiclass_classifier':
        num_classes = None

    aws_access_key_id = connection['accessKeyId']
    aws_secret_access_key = connection['secretAccessKey']

    boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)

    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sagemaker_client = sagemaker_session.sagemaker_client

    sagemaker_linear_learner = LinearLearner(predictor_type=predictor_type,
                                             role=role,
                                             num_classes=num_classes,
                                             train_instance_count=train_instance_count,
                                             train_instance_type=train_instance_type,
                                             train_volume_size=train_volume_size,
                                             train_max_run=train_max_run,
#                                             binary_classifier_model_selection_criteria=binary_classifier_model_selection_criteria,
#                                             target_recall=target_recall,
#                                             target_precision=target_precision,
#                                             positive_example_weight_mult=positive_example_weight_mult,
#                                             epochs=epochs,
#                                             use_bias=use_bias,
#                                             num_models=num_models,
#                                             num_calibration_samples=num_calibration_samples,
#                                             init_method=init_method,
#                                             init_scale=init_scale,
#                                             init_sigma=init_sigma,
#                                             init_bias=init_bias,
#                                             optimizer=optimizer,
#                                             loss=loss,
#                                             wd=wd,
#                                             l1=l1,
#                                             momentum=momentum,
#                                             learning_rate=learning_rate,
#                                             beta_1=beta_1,
#                                             beta_2=beta_2,
#                                             bias_lr_mult=bias_lr_mult,
#                                             bias_wd_mult=bias_wd_mult,
#                                             use_lr_scheduler=use_lr_scheduler,
#                                             lr_scheduler_step=lr_scheduler_step,
#                                             lr_scheduler_factor=lr_scheduler_factor,
#                                             lr_scheduler_minimum_lr=lr_scheduler_minimum_lr,
#                                             normalize_data=normalize_data,
#                                             normalize_label=normalize_label,
#                                             unbias_data=unbias_data,
#                                             unbias_label=unbias_label,
#                                             num_point_for_scaler=num_point_for_scaler,
#                                             margin=margin,
#                                             quantile=quantile,
#                                             loss_insensitivity=loss_insensitivity,
#                                             huber_delta=huber_delta,
#                                             early_stopping_patience=early_stopping_patience,
#                                             early_stopping_tolerance=early_stopping_tolerance,
#                                             accuracy_top_k=accuracy_top_k,
#                                             f_beta=f_beta,
#                                             balance_multiclass_weights=balance_multiclass_weights,
                                             sagemaker_session=sagemaker_session,
                                             **kwargs)

    try:
        sagemaker_linear_learner.fit(sagemaker_linear_learner.record_set(features_arr, label_arr),
                        mini_batch_size=mini_batch_size,
                        wait=wait, job_name=job_name)
    finally:
        if sagemaker_linear_learner.latest_training_job is not None:
            sagemaker_client = sagemaker_session.sagemaker_client

            job_name = sagemaker_linear_learner.latest_training_job.name
            model_data = get_model_data_path(sagemaker_linear_learner)
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
            | ## Sagemaker Linear Learner Result
            | {message}
            |
            | ### Logs
            | ```
            | {logs}
            | ```'''.format(message=message, logs=logs)))

            model = _model_dict('sagemaker.linear_learner_train')
            model['feature_cols'] = feature_cols
            model['label_col'] = label_col
            model['predictor_type'] = predictor_type
            model['region_name'] = region_name
            model['role'] = role
            model['model_data'] = model_data
            model['description'] = description
            model['_repr_brtc_'] = rb.get()

            return {'model':model}


def linear_learner_predict(
    table, model,
    connection,
    prediction_col='prediction',
    instance_count=1, instance_type='ml.m5.large'):

    if model['_context'] == 'python' and model['_type'] == 'sagemaker.linear_learner_train':
        aws_access_key_id = connection['accessKeyId']
        aws_secret_access_key = connection['secretAccessKey']

        region_name = model['region_name']
        model_path = model['model_data']
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

        linear_learner_model = sagemaker.LinearLearnerModel(model_data=model_path,
                                                            role=role,
                                                            sagemaker_session=sagemaker_session)
        transformer = linear_learner_model.transformer(instance_count=instance_count,
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

    return {'out_table': out_table}
