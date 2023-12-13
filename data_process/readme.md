# Get data from GSE 
1. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742 
2. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138

We use the data from level 2 (counts of gene expression)
'''
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138%5FBroad%5FLINCS%5FLevel2%5FGEX%5Fn345976x978%5F2017%2D03%2D06%2Egctx%2Egz 

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138%5FBroad%5FLINCS%5Finst%5Finfo%5F2017%2D03%2D06%2Etxt%2Egz

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138%5FBroad%5FLINCS%5Fgene%5Finfo%5F2017%2D03%2D06%2Etxt%2Egz 
'''

# Get differential gene expression by using Limma 


