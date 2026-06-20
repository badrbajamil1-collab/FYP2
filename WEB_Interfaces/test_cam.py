import urllib.request, re
req = urllib.request.Request('https://opencctv.org/cam/328591', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode()
with open('opencctv.html', 'w', encoding='utf-8') as f:
    f.write(html)
