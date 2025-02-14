# ベースイメージとして公式のPythonイメージを使用
FROM python:3.12.1-bullseye

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /workspace

# ボリュームを定義（必要に応じて）
VOLUME /workspace/data

# ユーザー設定（VS Code Remote - Containersで推奨される設定）
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# PYTHONPATHの設定
ENV PYTHONPATH=/workspaces/container-test:$PYTHONPATH

# 非ルートユーザーに切り替え
USER $USERNAME