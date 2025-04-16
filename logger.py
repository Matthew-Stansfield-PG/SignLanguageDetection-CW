import wandb
import yaml

class Logger:
    def __init__(self, experiment_name, logger_name='logger', project='INM705-COURSEWORK-Tuning'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger_name = f'{logger_name}-{experiment_name}'
        logger = wandb.init(project=project, name=logger_name, config=config)
        self.logger = logger
        return

    def get_logger(self):
        return self.logger

#logger=Logger(experiment_name='test')

# class Logger:
#
#     def __init__(self, experiment_name, logger_name='logger', project='INM705-COURSEWORK'):
#         logger = wandb.init(project=project, job_type="training", name=logger_name)
#         self.logger = logger
#         return
#
#     def get_logger(self):
#         return self.logger