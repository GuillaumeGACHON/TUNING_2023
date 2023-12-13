if [[ -f "Climatos_ATM_LR.nc" ]]; then
    echo "Climatos_ATM_LR.nc found"
else
    echo "Downloading Climatos_ATM_LR.nc"
    curl -o Climatos_ATM_LR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_ATM_LR.nc"
fi 

if [[ -f "Climatos_ATM_VLR.nc" ]]; then
    echo "Climatos_ATM_VLR.nc found"
else
    echo "Downloading Climatos_ATM_VLR.nc"
    curl -o Climatos_ATM_VLR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_ATM_VLR.nc"
fi

if [[ -f "Climatos_ICE_LR.nc" ]]; then
    echo "Climatos_ICE_LR.nc found"
else
    echo "Downloading Climatos_ICE_LR.nc"
    curl -o Climatos_ICE_LR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_ICE_LR.nc"
fi

if [[ -f "Climatos_ICE_VLR.nc" ]]; then
    echo "Climatos_ICE_VLR.nc found"
else
    echo "Downloading Climatos_ICE_VLR.nc"
    curl -o Climatos_ICE_VLR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_ICE_VLR.nc"
fi

if [[ -f "Climatos_OCE_Grid_T_LR.nc" ]]; then
    echo "Climatos_OCE_Grid_T_LR.nc found"
else
    echo "Downloading Climatos_OCE_Grid_T_LR.nc"
    curl -o Climatos_OCE_Grid_T_LR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_OCE_Grid_T_LR.nc"
fi

if [[ -f "Climatos_OCE_Grid_T_VLR.nc" ]]; then
    echo "Climatos_OCE_Grid_T_VLR.nc found"
else
    echo "Downloading Climatos_OCE_Grid_T_VLR.nc"
    curl -o Climatos_OCE_Grid_T_VLR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_OCE_Grid_T_VLR.nc"
fi

if [[ -f "Climatos_OCE_Diaptr_W_LR.nc" ]]; then
    echo "Climatos_OCE_Diaptr_W_LR.nc found"
else
    echo "Downloading Climatos_OCE_Diaptr_W_LR.nc"
    curl -o Climatos_OCE_Diaptr_W_LR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_OCE_Diaptr_W_LR.nc"
fi

if [[ -f "Climatos_OCE_Diaptr_W_VLR.nc" ]]; then
    echo "Climatos_OCE_Diaptr_W_VLR.nc found"
else
    echo "Downloading Climatos_OCE_Diaptr_W_VLR.nc"
    curl -o Climatos_OCE_Diaptr_W_VLR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_OCE_Diaptr_W_VLR.nc"i
fi

if [[ -f "Climatos_OCE_Grid_T_DepthLv_LR.nc" ]]; then
    echo "Climatos_OCE_Grid_T_DepthLv_LR.nc found"
else
    echo "Downloading Climatos_OCE_Grid_T_DepthLv_LR.nc"
    curl -o Climatos_OCE_Grid_T_DepthLv_LR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_OCE_Grid_T_DepthLv_LR.nc"
fi

if [[ -f "Climatos_OCE_Grid_T_DepthLv_VLR.nc" ]]; then
    echo "Climatos_OCE_Grid_T_DepthLv_VLR.nc found"
else
    echo "Climatos_OCE_Grid_T_DepthLv_VLR.nc found"
    curl -o Climatos_OCE_Grid_T_DepthLv_VLR.nc "https://thredds-su.ipsl.fr/thredds/fileServer/tgcc_thredds/work/gachongu/Tuning_2023/Climatos_OCE_Grid_T_DepthLv_VLR.nc"
fi
