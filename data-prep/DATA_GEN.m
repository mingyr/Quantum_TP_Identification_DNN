% "Quantum topology identification  with deep neural networks and quantum
% walks"  published on NJP Computational Materials 


%function evolution_topo(m,t1y,t2,t3,N,T1)

% The hamiltonian of our model is H = 2*t1x*cos(kx)*sigma_x 
% + 2*t1y*cos(ky)*sigma_y + {m + 2*t2*cos(kx+ky) +
% 2*t3*(sin(kx)+sin(ky))} * sigma_z;

% N is the size of the lattice we are simulating.  
% here  we set t1x=t1y=1

% T1 is the evoultion duration.


tt=0; %% tt is the strength of a third nearest neighbour (hopping) term in
% x direction; and tt=3,6,9 are simulated in our work.

t1y = 1;
t2=5;
N=599;
Nx = N;
Ny = N;
t1x = 1;
kx=linspace(0,2*pi*(Nx-1)/Nx,Nx);
ky=linspace(0,2*pi*(Ny-1)/Ny,Ny);

%define the delta intial state
Gx=ones(1,Nx)/sqrt(Nx);
Gy=ones(1,Ny)/sqrt(Ny);



m0 = linspace(-20,20,56);
t30 = linspace(-20,20,56);

for i = 1 : length(m0)
    m = m0(i);
    for j = 1 : length(t30)
        t3 = t30(j);
        
        if  abs(t3)>18
            T1 = 6;          
        elseif abs(t3)>12
            T1 = 8;
        elseif abs(t3)>10
            T1 = 10;          
        elseif abs(t3)>8
            T1 = 12;           
        elseif abs(t3)>6
            T1 = 5;          
        elseif abs(t3)>5
            T1 = 16;
        elseif abs(t3)>4.25
            T1 = 22;          
        elseif abs(t3)>3.5
            T1 = 30;
        elseif abs(t3)>2.5
            T1 = 35;
        elseif abs(t3)>2.25
            T1 = 40;
        elseif abs(t3)>2
            T1 = 45;   
        elseif abs(t3)>1.75
            T1 = 50;
        elseif abs(t3)>1.5
            T1 = 55;
        elseif abs(t3)>1.25
            T1 = 60;   
        elseif abs(t3)>1
            T1 = 65;             
        elseif abs(t3)>0.75
            T1 = 70;
        elseif abs(t3)>0.5
            T1 = 83;
        elseif abs(t3)>0.25
            T1 = 100;
        elseif abs(t3)<0.25
            T1 = 110;
        end
        
        T1=T1-1.5;
        
        for x= 1:Nx
            for y= 1:Ny
                sita = T1*sqrt((2*t1x*cos(kx(x)))^2+(2*t1y*cos(ky(y)))^2+(m+2*t2*cos(kx(x)+ky(y))+1.5*t3*(sin(kx(x))+sin(ky(y)))+tt*cos(2*kx(x)))^2);
                f1(x,y)=Gx(x)*Gy(y)*exp(1i*((Nx-1)/2*kx(x)+(Ny-1)/2*ky(y)))*(cos(sita)-1i*sin(sita)/sita*T1*(m+2*t2*cos(kx(x)+ky(y))+1.5*t3*(sin(kx(x))+sin(ky(y)))+tt*cos(2*kx(x))));
                f2(x,y)=Gx(x)*Gy(y)*exp(1i*((Nx-1)/2*kx(x)+(Ny-1)/2*ky(y)))*(-1i)*sin(sita)/sita*T1*(2*t1x*cos(kx(x))+2*t1y*cos(ky(y))*1i);
            end
        end
       
       kup = abs(fft2(f1)).^2/(Nx*Ny);
       kdown = abs(fft2(f2)).^2/(Nx*Ny);
       
       file_name = sprintf('m_%2.2f_t1y_%d_t2_%2.1f_t3_%2.2f.mat',m,t1y,t2,t3);    
       save(['data' file_name],'f1','f2','kup','kdown');
        
    end
end


% this part is adding noise to the data.
sigma=[1,5,11,23,47]; %gaussian noise 
  for ss = 1:length(sigma)
        sig=sigma(ss);
        f1_noise=imgaussfilt(real(f1), sig)+1i*imgaussfilt(imag(f1), sig);
        f2_noise=imgaussfilt(real(f2), sig)+1i*imgaussfilt(imag(f2), sig);   
        kup_noise=imgaussfilt(real(kup), sig)+1i*imgaussfilt(imag(kup), sig);
        kdown_noise=imgaussfilt(real(kdown), sig)+1i*imgaussfilt(imag(kdown), sig);  
        
        f1_noise= imnoise(real(f1_noise),'poisson')*abs(max(max(real(f1))))+1i*imnoise(imag(f1_noise),'poisson')*abs(max(max(imag(f1))));
        f2_noise= imnoise(real(f2_noise),'poisson')*abs(max(max(real(f2))))+1i*imnoise(imag(f2_noise),'poisson')*abs(max(max(imag(f1)))); 
        kup_noise= imnoise(real(kup_noise),'poisson')+1i*imnoise(imag(kup_noise),'poisson');
        kdown_noise= imnoise(real(kdown_noise),'poisson')+1i*imnoise(imag(kdown_noise),'poisson');     
        
        save(['noise_data' file_name],'f1_noise','f2_noise','kup_noise','kdown_noise');      
 end

