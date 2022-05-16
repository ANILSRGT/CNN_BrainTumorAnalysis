import os

from brain_tumor_training_multi_feature import train_run as multi_train_run
from brain_tumor_training_single_feature import train_run as single_train_run


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    multi_train_run()
#    while True:
#        print("""
#            0 - RUN Single Feature Train
#            1 - RUN Multi Feature Train
#            q - Exit
#            """)
#        choose = input("Seçim yapınız : ")
#        if choose == "0":
#            single_train_run()
#            input("\nDevam etmek için tıklayınız...")
#        elif choose == "1":
#            multi_train_run()
#            input("\nDevam etmek için tıklayınız...")
#        elif choose == 'q' or choose == "Q":
#            cls()
#            print("Çıkış yapılıyor...")
#            exit(0)
#        else:
#            cls()
#            print("Hatalı Giriş Yapıldı!")
#
