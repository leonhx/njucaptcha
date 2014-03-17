南京大学统一身份认证平台验证码识别系统
==========

http://cer.nju.edu.cn/amserver/UI/Login

注意：由于南京大学统一身份认证平台随时有可能提高验证码识别难度，所以此系统无法保证能一直正常使用。下面的`数据更新时间`即最近一次训练识别系统的时间，如果在此时间之后验证码识别难度增加，则无法正常识别。

数据更新时间：2014年3月10日20:02:07

此系统预测依赖于`scikit-learn`、`PythonCV`、`numpy`包，请先行安装。


Example
=======

```python
import recognizer

page = urllib2.urlopen('http://cer.nju.edu.cn/amserver/verify/image.jsp')
captcha_str = page.read()

result = predict(clean_captcha(get_captcha(captcha_str))) # 这就是识别结果了
```
