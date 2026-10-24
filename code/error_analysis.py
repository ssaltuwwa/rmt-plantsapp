"""Error analysis of PlantDoc test predictions: top confusions, prediction collapse, stock-photo subset."""
import pandas as pd, csv
n={int(r['class_id']):r['plantdoc'] for r in csv.DictReader(open('../data_out/class_map.csv'))}
t=pd.read_csv('../data_out/pd_manifest.csv'); t=t[t.split=='test']
k=t.source_image.str.lower().str.contains('isolated|white-background|on-white')
print('test crops from "isolated on white" stock photos:',k.sum(),'of',len(t),'; images:',t[k].source_image.nunique(),'of',t.source_image.nunique())
for r in ['baseline_s42','ft50_s42']:
    p=pd.read_csv(f'../runs/{r}/plantdoc_test_predictions.csv'); w=p[p.pred!=p.label]
    print(f'\n{r}: errors {len(w)} of {len(p)}'); print((w.label.map(n)+' -> '+w.pred.map(n)).value_counts().head(6).to_string())
    print('most frequent predicted classes:', p.pred.map(n).value_counts().head(3).to_dict())
    p['iso']=p.path.str.lower().str.contains('isolated|white-background|on-white')
    print('accuracy on stock "isolated" crops:',round((p[p.iso].pred==p[p.iso].label).mean(),3),'| other crops:',round((p[~p.iso].pred==p[~p.iso].label).mean(),3))
b=pd.read_csv('../runs/baseline_s42/plantdoc_test_per_class.csv'); f=pd.read_csv('../runs/ft50_s42/plantdoc_test_per_class.csv')
b['name']=b.label.map(n); b['f1_ft50']=f.f1; b['delta']=b.f1_ft50-b.f1
b.sort_values('f1_ft50')[['label','name','support','f1','f1_ft50','delta']].round(3).to_csv('../runs/per_class_baseline_vs_ft50.csv',index=False)
print('\nclasses with F1 = 0: baseline',(b.f1==0).sum(),'| ft50',(f.f1==0).sum())
