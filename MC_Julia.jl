using PyPlot

##################### calculation parameters ##############
Lx =10   ##system size in the x-direction
Ly =10   ##system size in the y-direction

J = -1

Bvec = [0.01; 0; 0.0]  #magnetic field

An = [0; 0; 0.0]       #Uniaxial anisotropy

T = 0.1  # temperature

maxMCstep = Lx * Ly * 1000   ### # of MC step


calctimes = 1  # How many times we run the MC


########################################################



function normalize_vect(randx,randy,spin_x, spin_y, spin_z)

     norm = sqrt(spin_x[randy, randx]^2 + spin_y[randy, randx]^2 + spin_z[randy, randx]^2)

     spin_x[randy, randx]/=norm
     spin_y[randy, randx]/=norm
     spin_z[randy, randx]/=norm

     return spin_x, spin_y, spin_z

end



function local_energy(x, y, spin_x,spin_y,spin_z)  #calculate the single site energy


    vec0 = [spin_x[y,x]; spin_y[y,x] ; spin_z[y,x]]
    vecr = [spin_x[y,x+1] ; spin_y[y,x+1]; spin_z[y,x+1]]
    vecl = [spin_x[y,x-1];  spin_y[y,x-1]; spin_z[y,x-1]]
    vecu = [spin_x[y-1,x]; spin_y[y-1,x]; spin_z[y-1,x]]
    vecd = [spin_x[y+1,x]; spin_y[y+1,x]; spin_z[y+1,x]]


     exch = J * sum(vec0 .* (vecr + vecl + vecu + vecd))
     zeeman = - vecdot(Bvec, vec0)
     anis =  - vecdot(vec0, vec0 .* An)

     return exch + zeeman + anis

end




function mc_step(energy, spin_x, spin_y, spin_z,T)  # local update MC with Metropolis alogrithm


    ####### local update #######

     randx = rand(2:Lx+1)
     randy = rand(2:Ly+1)

     locen = copy(local_energy(randx,randy,spin_x,spin_y,spin_z))

     temp = copy([spin_x[randy,randx] spin_y[randy,randx] spin_z[randy,randx]])

     spin_x[randy, randx] += (2 *rand() -1)
     spin_y[randy, randx] +=  (2 *rand() -1)
     spin_z[randy, randx] += (2 *rand() -1)

     spin_x, spin_y, spin_z = normalize_vect(randx,randy,spin_x,spin_y,spin_z)

    ################ calculate the energy of the new spin config #######

     de = - locen + local_energy(randx,randy,spin_x,spin_y,spin_z)

    ############### Metropolis ##############
     if rand() > exp(-de/T)
            spin_x[randy,randx],  spin_y[randy,randx],  spin_z[randy, randx] = temp
     end

     return spin_x, spin_y, spin_z

end



@parallel for seedval in 1:calctimes      # Multiple calculations in a paralellized way

    srand(seedval)


    ######### initialize the spin configuration #######
    spin_x = (rand(Ly+2,Lx+2)) * 0 + 1
    spin_y = (2*rand(Ly+2,Lx+2)-1) * 0
    spin_z = (2*rand(Ly+2,Lx+2)-1) * 0

    norm = sqrt(spin_x.^2 + spin_y.^2 + spin_z.^2)

    spin_x./=norm
    spin_y./=norm
    spin_z./=norm
    #################################################





    #####################   impose the open boundary condition ###################


    box = zeros(Ly+2,Lx+2)

    box[2:Ly+1,2:Lx+1] = ones(Ly,Lx)

    spin_x = spin_x .*box
    spin_y = spin_y .*box
    spin_z = spin_z .*box

    #########################################################################

    energy = 0   # we don't need the total energy for MC (as long as using the local update scheme)

    meanspinx = []
    meanspiny = []
    meanspinz = []

    count = 0

    ########################## MC run  ######################################
    for i in 1:maxMCstep

        spin_x, spin_y, spin_z = mc_step(energy, spin_x, spin_y, spin_z, T)


        if i % (Lx*Ly) == 0
            push!(meanspinx, sum(spin_x)/Lx/Ly)
            push!(meanspiny, sum(spin_y)/Lx/Ly)
            push!(meanspinz, sum(spin_z)/Lx/Ly)
            count += 1
        end


    end

####################### plot spin ######################################

    plot((1:(count)) * Lx*Ly,meanspinx)
    plot((1:(count)) * Lx*Ly,meanspiny)
    plot((1:(count)) * Lx*Ly,meanspinz)

end


