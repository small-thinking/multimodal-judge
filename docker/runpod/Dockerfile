FROM smallthinking/multimodal-judge@sha256:b45e9cdfdbc9171c0921ff09454f678e515c2944a34553650e106522b5347807
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server git tmux && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_*
COPY start.sh /usr/local/bin/runpod-start
RUN chmod 755 /usr/local/bin/runpod-start
ENV HF_HOME=/workspace/cache/huggingface
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/runpod-start"]
CMD []
