Ubuntu系统图形界面卡住：

```
sudo pkill Xorg

```

xfce4-panel -q
mv ~/.config/xfce4/panel ~/.config/xfce4/panel.bak 2>/dev/null
mv ~/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml \
   ~/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml.bak 2>/dev/null
xfce4-panel &