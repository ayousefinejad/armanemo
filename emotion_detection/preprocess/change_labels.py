
def change_label_name_new_dataset(df, label):
    df[label].replace(['ANGRY', 'HATE', 'FEAR'], 'عصبانی', inplace=True)
    df[label].replace(['HAPPY'], 'عاشقانه و خوشحال', inplace=True)
    df[label].replace(['SAD'], 'غمگین و مضطرب', inplace=True)
    df[label].replace(['SURPRISE'], 'هیجانی و متعجب', inplace=True)
    df = df[~df[label].isin(['OTHER'])]
    return df


def label_ecoding(df, label):
    df[label].replace([' عاشقانه', 'عاشقانه', 'عاشقانه ', 'خوشال','خوشحال','خوشحال ','خوشحالی'], ' عاشقانه و خوشحال', inplace=True)
    df[label].replace(['هییجان انگیز ',  'هیجانزده',  'هیجان زده ', 'هیجان زده', 'هیجان انگیز ', 'هیجان انگیز', 'هیجان',
                        'شگفت زده', 'تعجب','متعجب','با تمام مشکلات خودمو کشیدم بالا و گوشمو تابوندم هر کسی رو محرم خودم ندونم+D23901:D23906'],
                        'هیجانی و متعجب', inplace=True)
    df[label].replace(['نارحت', 'ناراحتی', 'ناراحت ', 'ناراحت', 'مناراحت', 'غمگین', 'مضطرب','اضطراب','نگران'], 'غمگین و مضطرب', inplace=True)
    df[label].replace(['معمول','معمولا','معمولب','معمولی','معمولی ','معولی','ممعمولی','ممولی'], 'معمولی', inplace=True)
    df[label].replace([ 'عصبانی','عصبانی ','اعصبانی','خشم', 'ترس','ترس ','ترسناک', 'اعتراض', 'اعتراضی', 'تنفر'], 'عصبانی', inplace=True)

    df = df[~df[label].isin(['رس','شرمندگی', 'نترس', 'اعتراضی'])]
    df.to_csv('data/preprocess_labelencoding_data.csv', index=False)
    return df