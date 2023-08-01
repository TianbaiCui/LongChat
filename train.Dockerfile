# https://github.com/orgs/pytorch/packages/container/package/pytorch-nightly
FROM ghcr.io/pytorch/pytorch-nightly:153afbd-cu11.8.0

WORKDIR /home/llm_training

COPY . .

RUN pip install .

RUN pytest tests/test_datamodule.py

ENTRYPOINT [ "deepspeed" ]
CMD ["longchat/train/fine_tune/train_adalora.py", "--config_file", "longchat/train/configs/train_config.yaml"]