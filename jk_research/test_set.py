from datetime import date

start_date = date(2023, 1, 1)
end_date = date(2023, 8, 1)

for day in range(start_date.toordinal(), end_date.toordinal() + 1):
    _date = date.fromordinal(day)
    print(f"python ~/openfold/scripts/download_cameo.py 1-year {_date} ~/tmp/cameoDL1yr/{_date}/")