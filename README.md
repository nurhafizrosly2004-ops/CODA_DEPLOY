# CODA-Kiosk — Fashion Detection Kiosk with PyQt5 & YOLOv8

> 🎯 Deployable on **RDK X5 (ARM64)** running **Ubuntu 22.04** — uses **PyQt5** for guaranteed compatibility.

This kiosk detects fashion items (formal, wedding, mosque) using YOLOv8 and displays a leaderboard with scores.

---

## 🛠️ Installation (On RDK X5 / Ubuntu 22.04)

### 1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/CODA-Kiosk.git
cd CODA-Kiosk

linux
sudo apt update
sudo apt install -y python3-pip libxcb-cursor0 libxcb-xinerama0 libx11-xcb1

pip3 install --user -r requirements.txt

if pip3 install pyQT5 fails:
sudo apt install python3-pyqt5