FROM python:3.8-buster as my_builder_stage
# Config PART pip
COPY sources.list /etc/apt/sources.list

RUN pip config set global.index http://pypi.partdp.ir/root/pypi &&\
    pip config set global.index-url http://pypi.partdp.ir/root/pypi/+simple/ &&\
    pip config set global.trusted-host pypi.partdp.ir


COPY . ./opt/devops/ai/emotion_detection/
WORKDIR ./opt/devops/ai/emotion_detection/

RUN apt update
RUN pip install --upgrade pip
RUN pip install ./dist/torch-1.6.0+cpu-cp38-cp38-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache  pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/opt/devops/ai/emotion_detection/"

# Second stage
FROM python:3.8-slim-buster
COPY sources.list /etc/apt/sources.list

COPY --from=my_builder_stage /usr/local/ /usr/local/
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONPATH "${PYTHONPATH}:/opt/devops/ai/emotion_detection/"

COPY --from=my_builder_stage /opt/devops/ai/emotion_detection/ /opt/devops/ai/emotion_detection/
COPY --from=my_builder_stage /root/ /root/

WORKDIR ./opt/devops/ai/emotion_detection/

CMD [ "/usr/local/bin/python", "/opt/devops/ai/emotion_detection/emotion_detection/app.py" ]
