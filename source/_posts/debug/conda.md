---
title: conda

date: 2024-6-14

tags: [conda,debug]

categories: [debug]

comment: false

toc: true

---

#

<!--more-->

# conda debug



## 1. 创建环境

- 问题：conda create时出现：

    ```bash
    Solving environment: failed

    CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/r/noarch/repodata.json.bz2>
    Elapsed: -

    An HTTP error occurred when trying to retrieve this URL.
    HTTP errors are often intermittent, and a simple retry will get you on your way.

    If your current network has https://www.anaconda.com blocked, please file
    a support request with your network engineering team.

    SSLError(MaxRetryError('HTTPSConnectionPool(host=\'repo.anaconda.com\', port=443): Max retries exceeded with url: /pkgs/r/noarch/repodata.json.bz2 (Caused by SSLError(SSLError("bad handshake: Error([(\'SSL routines\', \'ssl3_get_server_certificate\', \'certificate verify failed\')])")))'))
    ```

- 解决：禁用SSL验证，作为临时解决方案，这样做可能会降低安全性。

    ```bash
    conda config --set ssl_verify no
    ```

