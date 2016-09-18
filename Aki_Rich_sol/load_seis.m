load seis.csv
labels=cellstr(char('Ux', 'Uy', 'Uz', 'Ur', 'Uth', 'Uphi')); 

for i=1:6
    subplot(3,3,i)
    plot(seis(1,:), seis(i+1,:))
    xlabel('time (s)')
    ylabel(labels(i))
end



