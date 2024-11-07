rm -v data/intermediate/*
rm -v data/result/*
find data/raw/ -type f -not \( -name 'token_list.txt' -or -name 'file2.png' \) -delete
find weight/ -type f -not \( -name 'mol_pre_all_h_220816.pt' -or -name 'mol_pre_no_h_220816.pt' \) -delete
