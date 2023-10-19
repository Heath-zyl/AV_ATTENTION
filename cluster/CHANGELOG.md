## 1.1.1 / 2022-09-27
-FIX: 修复pyface.conf中mounts的格式问题  
## 1.1 / 2022-09-26
-FIX: 修复多机训练偶尔启动失败的问题。原因是pod 0的地址总是在pod创建好之后过一定的时间才会生效，解决方式是增加init container循环ping pod 0地址，直到ping通   
-MOD: 支持mount路径可配置  
## 1.0 / 2022-09-20
-ADD: 支持cluster.luanch接口  
-ADD: 支持原生kubernetes部署(scheduler:raw)  
-ADD: 支持volcano部署（scheduler:torchx)  
-ADD: 支持ray部署（scheduler:ray)  
