xSM = xSMplus;
vxSM = vxSMplus;
ySM = ySMplus;
vySM = vySMplus;
tS = tSplus;
d = size(tS);
d = d(2);
sizes = size(xSM);
for i = 15
    t = tS(i); %this is smallest time
    X = reshape(xSM(i,:,:,:),[],1);
    Y = reshape(ySM(i,:,:,:),[],1);
    VX = reshape(vxSM(i,:,:,:),[],1);
    VY = reshape(vySM(i,:,:,:),[],1);
    figure
    scatter3(X(:),Y(:),VY(:),'filled','b','DisplayName', 'Stable Manifold');
    hold on
    scatter3(Xhyperbolicplus(i,1),Xhyperbolicplus(i,2),Xhyperbolicplus(i,4),100,'r','filled','DisplayName', 'Hyperbolic trajectory' );
    for j =1 : sizes(2)
        for k = 1: sizes(3)
            plot3(reshape(xSM(i,j,k,:),[],1),reshape(ySM(i,j,k,:),[],1),reshape(vySM(i,j,k,:),[],1),'b')
            hold on
        end
    end
    for j =1 : sizes(3)
        for k = 1: sizes(4)
            plot3(reshape(xSM(i,:,j,k),[],1),reshape(ySM(i,:,j,k),[],1),reshape(vySM(i,:,j,k),[],1),'g')
            hold on
        end
    end
    for j =1 : sizes(2)
        for k = 1: sizes(4)
            plot3(reshape(xSM(i,j,:,k),[],1),reshape(ySM(i,j,:,k),[],1),reshape(vySM(i,j,:,k),[],1),'y')
            hold on
        end
    end
    legend('Stable Manifold','Positive hyperbolic trajectory')

    figtitle = "Periodic system stable manifold points in (x,y,v_y) space at t = " + string(round(t,2));
    title(figtitle);
    xlabel("x");
    ylabel("y");
    zlabel("v_y");  
    savefig(figtitle+".fig");
    hold off
    figure
    scatter3(X(:),Y(:),VX(:),'filled','b','DisplayName', 'Stable Manifold');
    hold on
    scatter3(Xhyperbolicplus(i,1),Xhyperbolicplus(i,2),Xhyperbolicplus(i,3),100,'r','filled','DisplayName', 'Hyperbolic trajectory' );
        for j =1 : sizes(2)
        for k = 1: sizes(3)
            plot3(reshape(xSM(i,j,k,:),[],1),reshape(ySM(i,j,k,:),[],1),reshape(vxSM(i,j,k,:),[],1),'b')
            hold on
        end
    end
    for j =1 : sizes(3)
        for k = 1: sizes(4)
            plot3(reshape(xSM(i,:,j,k),[],1),reshape(ySM(i,:,j,k),[],1),reshape(vxSM(i,:,j,k),[],1),'g')
            hold on
        end
    end
    for j =1 : sizes(2)
        for k = 1: sizes(4)
            plot3(reshape(xSM(i,j,:,k),[],1),reshape(ySM(i,j,:,k),[],1),reshape(vxSM(i,j,:,k),[],1),'y')
            hold on
        end
    end
    legend('Stable Manifold','Positive hyperbolic trajectory')
    figtitle = "Periodic system stable manifold points in (x,y,v_x) space at t = " + string(round(t,2));
    title(figtitle);
    xlabel("x");
    ylabel("y");
    zlabel("v_x");  
    savefig(figtitle+".fig");
    hold off
    hold off

end

