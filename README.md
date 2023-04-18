# README
## 執行
首先需要建立一個input file的list  
格式如下:  
./example1.file time1  
./example2.file time2  
...  
其中第一項為照片的路徑(絕對路徑或相對於hw1.py的路徑)  
第二項為曝光時間的倒數  
如下圖:  
![](https://i.imgur.com/ShdxOpr.png)  
接下來執行  
```shell
python3 hw1.py < list.txt
```
執行時間大約數分鐘至數十分鐘不等  
## 輸出

輸出為:  
/data/HDR_img.hdr 為本次作業所需hdr輸出檔  
/data/Ldr_global.png 為使用photographic global tone mapping的LDR image  
/data/Ldr_local.png 為使用photographic local tone mapping的LDR image  
/data/Ldr_log.png 為使用adaptive logarithmic mapping的LDR image  
/curve/Curve_0.png  
/curve/Curve_1.png  
/curve/Curve_2.png 為BGR的response curve結果  
## package
本次使用的python額外套件  
cv2  
numpy  
matplotlib  

