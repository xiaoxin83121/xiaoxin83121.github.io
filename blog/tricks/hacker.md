# Some Tools helping you in the disguise of PRO?

> 一些让你看起来比较pro的工具的使用， 自用

## 浏览器

> 肯定是选择大家都用的chromium内核的，比如chrome和巨硬强制的chromium内核edge;优势是插件和开发者工具。

- **vimium**：在chrome的商店里就能下载(vpn)，和Unix中的vim编辑器操作类似。
```
f: 在当前页面打开页面中的所有link，current pages；
F(shift+f):在新标签页打开页面中的link，new pages；
j：向下滚动（少）；d:向下滚动（多）；
k: 向上滚动（少）；u:向上滚动（多）；
G(shift+G):到底部；gg:到顶部
J(shift+j):移动到左边的标签页；K(shift+k):移动到右边的标签页；
x:关闭当前标签页；X(shift+x):打开上一个被关闭的标签页；
o:current打开新的搜索结果；O(shift+o):new page打开新的搜索结果；
```
需要特别注意的是：遇到一些包含输入框的网页，ESC键可以帮助取消focus状态，和vim一致；  
以及，o操作默认的是google，调整成百度要在vimium->选项->custome search engines把
```
bd: https://www.baidu.com/s?wd=%s Baidu
```
写在第一行

- 油猴Tampermonkey
过于pro，自己写js不是傻瓜操作，直接去找现成的

- **ADBP： AdBlock Plus**

chrome商店下载即可，自动配置，再也不用看百度、csdn的秃头广告和盗版网站的H广告了  

- 开发者工具

F12打开，里头的东西需要一点前后端和网络知识，不表


## Markdown & latex
真正的写文字工具，论文只要有模板就不用考虑调格式的latex爽爆了（不转word的情况下），表格图片代码公式一应俱全的markdown爽爆了。
- 标题，***<u>字体</u>***等等省略
- **公式**
latex和markdown的公式相通，latex更丰富一些，markdown则能够在平时更简单轻松的写一些公式。markdown公式表示要用$content$包起来
```
$A_{i}^{j}$
$\frac{1+\aplha}{1 \times \aplha}$
$a \space b$
```
$A_{i}^{j}$  
$\frac{1+\alpha}{1 \times \alpha}$  
$a \space b$  
以上一一对应，抛砖引玉

- **图片**，建议使用的编辑器**Typora**,非常好用，分成所写即所得的编辑模式和源代码模式。最新的版本集成了picgo引擎，在**编辑模式**直接粘贴图片可以上传到SM.MS或者一些奇奇怪怪的图床里，教程也是现搜的（https://blog.csdn.net/bruce_6/article/details/104821531）实测可以下载picgo软件不配置命令行。

## Terminal

> 要Pro一定要命令行，Linux/Mac原生支持，Windows建议上巨硬应用商店里的Windows Terminal
> 当然，先问问自己ls、ps、grep是干啥的，不知道可以不用看
> 
- Windows仿Linux环境
配置Cygwin,下好了傻瓜安装就好了（可能要加一下系统环境变量，同时在安装过程中要根据自己需求安装哪些软件组 https://blog.csdn.net/lvsehaiyang1993/article/details/81027399 ），自带的cygwin terminal可以不用；  
之所以是仿linux，可以实现ssh，vim之类基础命令，Linux的部分无法实现；  
据说windows的preview里，内置了linux的内核，但是2020/9/4日仍然是1909版本，懂得抖动

- **screen**

又是一大利器，比tmux要方便得多，自行配置之后很舒服， 本质是为了解决在remote 连接服务器中，自身的网络或者服务器的网络导致连接中断时，Linux的SignUp机制会同时丢掉正在进行的服务，很伤很绝望  
先看看服务器有没有装screen，没装也没有root权限就byebye  

下面是几个入门常用的指令,| 后是实际示例
```
screen -S <session-name> | screen -S demo
screen -r <session-name> | screen -r demo
screen -ls
output: 
	12345.demo xxxxxx xxx xxx(Detached)
screen -X -S <session-id> quit | screen -X -S 12345 quit
```
当我们进入screen里，就是真正看起来pro的地方了；  
首先，在/home目录下新建~/.screenrc  ；  
粘贴以下内容：

```
# Set default encoding using utf8
defutf8 on


## 解决中文乱码,这个要按需配置
defencoding utf8
encoding utf8 utf8
 

#兼容shell 使得.bashrc .profile /etc/profile等里面的别名等设置生效
shell -$SHELL

#set the startup message
startup_message off
term linux

## 解决无法滚动
termcapinfo xterm|xterms|xs ti@:te=\E[2J
 
# 屏幕缓冲区行数
defscrollback 10000
 
# 下标签设置
hardstatus on
caption always "%{= kw}%-w%{= kG}%{+b}[%n %t]%{-b}%{= kw}%+w %=%d %M %0c %{g}%H%{-}"
 
#关闭闪屏
vbell off
 
#Keboard binding
# bind Alt+z to move to previous window
bindkey ^[z prev
# bind Alt+x to move to next window
bindkey ^[x next

# bind Alt`~= to screen0~12
bindkey "^[`" select 0
bindkey "^[1" select 1
bindkey "^[2" select 2
bindkey "^[3" select 3
bindkey "^[4" select 4
bindkey "^[5" select 5
bindkey "^[6" select 6
bindkey "^[7" select 7
bindkey "^[8" select 8
bindkey "^[9" select 9
bindkey "^[0" select 10
bindkey "^[-" select 11
bindkey "^[=" select 12
# bind F5 to create a new screen
bindkey -k k5 screen
# bind F6 to detach screen session (to background)
bindkey -k k6 detach
# bind F7 to kill current screen window
bindkey -k k7 kill
# bind F8 to rename current screen window
bindkey -k k8 title
```
结束后，**~/.screenrc不用也没办法source激活，但是在文件保存之前创建的session仍是原配置，所以想试一下效果需要保存之后重新创建session**；  
按F5之后，屏幕下方会出现[0 bash] [1 bash] 这类，按alt+z或者alt+x可以向左和向右调整，可以理解为浏览器的标签页；alt+{~-=}是切换标签页；  
F6会回退到进入screen的终端环境，再次进入使用screen -r <session-name>;     
我自己根据自己的习惯，改了改快捷操作和键位，增加了remove, focus(好像有bug), split-v, 具体快捷键的对应关系查看文档（https://www.gnu.org/software/screen/manual/screen.html#Default-Key-Bindings）中的5.1节



# Conclusion

祝咱都能很PRO