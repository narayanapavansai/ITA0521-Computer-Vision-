a=imread(r"C:\Users\SARAN\OneDrive\Pictures\HD-wallpaper-owl-eyes-owl-eyes-dark-staring-bird-black.jpg");
Lap=[0, 1, 0, 1, -4, 1, 0, 1, 0]; 
a1=conv2(a,Lap,r"C:\Users\SARAN\OneDrive\Pictures\HD-wallpaper-owl-eyes-owl-eyes-dark-staring-bird-black.jpg");
a2=uint8(a1);
imtool(abs(a-a2),[]) 
lap=[-1 ,-1, -1, -1, 8, -1, -1, -1 ,-1]; 
a3=conv2(a,lap,r"C:\Users\SARAN\OneDrive\Pictures\HD-wallpaper-owl-eyes-owl-eyes-dark-staring-bird-black.jpg");
a4=uint8(a3);
imtool(abs(a+a4),[]) 
