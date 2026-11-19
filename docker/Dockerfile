FROM ubuntu:22.04
LABEL maintainer="Minhyun Cho <cho515@purdue.edu>"

# -----------------------------------------------------------------------------
# Environment
# - environment variables to set pacakge manager
# - environment variable to designate the directory 
#   where temporary files and sockets are stored and accessible
# - environment variables to enable the use of NVIDIA GPU
#   ref. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
# - add bin folder as environmental variable
# - protobuff related environmental variable
# - environment variables for system locale
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York \
    XDG_RUNTIME_DIR=/tmp/runtime-docker \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PATH=/home/user/bin:${PATH} \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# Shell configuration
# - use bash instead of sh
# - enable pipefail to catch errors in piped commands
# -----------------------------------------------------------------------------
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# Build arguments
# - UID/GID for container user (to match host user if needed)
# - additional groups for device access (input, render)
# - host user/group info (mainly for metadata / consistency with compose)
# -----------------------------------------------------------------------------
ARG UID_USER=1000
ARG GID_USER=1000
ARG GID_INPUT=107
ARG GID_RENDER=110

ARG HOST_USER_NAME=user
ARG HOST_USER_ID=1000
ARG HOST_GROUP_NAME=user
ARG HOST_GROUP_ID=1000

ENV HOST_USER_NAME=${HOST_USER_NAME} \
    HOST_USER_ID=${HOST_USER_ID} \
    HOST_GROUP_NAME=${HOST_GROUP_NAME} \
    HOST_GROUP_ID=${HOST_GROUP_ID}

# -----------------------------------------------------------------------------
# Base packages and locale setup
# - set frontend as noninteractive mode
# - install minimal required packages (sudo, locales)
# - configure system locale to UTF-8
# - remove apt cache to reduce image size
# -----------------------------------------------------------------------------
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        locales \
        ca-certificates \
        bash \
        git \
        curl \
        wget \
        gnupg2 \
        lsb-release \
        software-properties-common \
        tzdata && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Groups and user setup
# - create additional system groups (input, render) for hardware/device access
# - create a non-root user inside the container
# - add user to commonly required groups (sudo, dialout, video, etc.)
# - configure passwordless sudo
# - create and set permissions for XDG runtime directory
# - setup environmental variable for container user to run dependencies installation
# -----------------------------------------------------------------------------
RUN if ! getent group input >/dev/null 2>&1; then groupadd -g "${GID_INPUT}" input; fi && \
    if ! getent group render >/dev/null 2>&1; then groupadd -g "${GID_RENDER}" render; fi && \
    groupadd --gid "${GID_USER}" user && \
    useradd --uid "${UID_USER}" --gid "${GID_USER}" --create-home --shell /bin/bash user && \
    usermod -aG sudo,plugdev,dialout,input,render,video user && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    mkdir -p "${XDG_RUNTIME_DIR}" && \
    chown user:user "${XDG_RUNTIME_DIR}" && \
    chmod 700 "${XDG_RUNTIME_DIR}"

ENV USER=user \
    HOME=/home/user

# -----------------------------------------------------------------------------
# Install scripts (copied with correct ownership for user execution)
# - base: common dependencies
# - ros2: ROS2 installation (e.g., Humble)
# - gazebo: simulation environment (Ignition Gazebo)
# - autopilot: PX4 / ArduPilot dependencies
# - extra: additional tools and utilities
# - usersetup: user-specific environment configuration
# -----------------------------------------------------------------------------
COPY --chown=user:user install/base.sh /tmp/install/base.sh
COPY --chown=user:user install/ros2.sh /tmp/install/ros2.sh
COPY --chown=user:user install/gazebo.sh /tmp/install/gazebo.sh
COPY --chown=user:user install/autopilot.sh /tmp/install/autopilot.sh
COPY --chown=user:user install/extra.sh /tmp/install/extra.sh
COPY --chown=user:user install/clean.sh /tmp/clean.sh

# make all install scripts executable
RUN chmod +x /tmp/install/*.sh

# -----------------------------------------------------------------------------
# Execute installation scripts as non-root user
# - ensures environment is configured similarly to real user environment
# - avoids permission issues in user workspace
# -----------------------------------------------------------------------------
USER user

RUN bash /tmp/install/base.sh
RUN bash /tmp/install/ros2.sh --ros-distro humble
RUN bash /tmp/install/gazebo.sh --install binary --phase deps
RUN bash /tmp/install/gazebo.sh --install binary --phase build
RUN bash /tmp/install/autopilot.sh --with-ardupilot --phase deps
RUN bash /tmp/install/extra.sh --phase deps

# switch back to root for final system-level operations
USER root

# -----------------------------------------------------------------------------
# Cleanup
# - run custom cleanup script
# - remove Docker-specific apt auto-clean config (can speed up future apt usage)
# -----------------------------------------------------------------------------
RUN bash /tmp/clean.sh && \
    rm -f /etc/apt/apt.conf.d/docker-clean || true

# -----------------------------------------------------------------------------
# Workspace setup
# - create standard working directories for PX4 and ROS2
# - assign ownership to non-root user
# -----------------------------------------------------------------------------
RUN mkdir -p /home/user/FIRE_flightstack_sim && \
    chown -R user:user /home/user/FIRE_flightstack_sim

# set working directory
WORKDIR /home/user/FIRE_flightstack_sim

# -----------------------------------------------------------------------------
# Container entrypoint
# - defines the default command when container starts
# -----------------------------------------------------------------------------
ENTRYPOINT ["./install/entrypoint.sh"]
CMD ["bash"]