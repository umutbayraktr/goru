from pytube import YouTube

# İndirmek istediğin videonun URL'si
url = 'https://www.youtube.com/watch?v=475NZ16m9BM'  # Buraya gerçek video linkini yaz

# YouTube nesnesi oluştur
yt = YouTube(url)

# En yüksek çözünürlüklü MP4 videosunu seç
video = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()

# Videoyu indir
video.download(output_path='indirilenler', filename='video.mp4')

print("İndirme tamamlandı.")
