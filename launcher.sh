#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate dash

echo " Welcome to the MeVisTo Launcher. Let’s check if you have the Climatos files and download them if necessary."
bash Climato_Checker.sh

echo " All done. Launching the App now."
python metrics_app_V2.2.py
