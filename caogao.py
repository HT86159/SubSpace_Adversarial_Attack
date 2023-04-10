def mim_subspace_init_attack(ori_image, eps, T, model, criterion, y_ture, beta, mu, device, Tpt, N):
    minx = ori_image - eps
    maxx = ori_image + eps
    image = ori_image
    temp = torch.rand_like(image).unsqueeze(0)
    image_shape = torch.tensor(temp.shape)
    image_shape[0] = N
    Delta = torch.randn(tuple(image_shape)) / Tpt
    Delta = Delta.to(device)
    # print(Delta.shape)

    g = 0
    alpha = eps / T
    mu = 1
    momentum = torch.zeros_like(Delta).detach().to(device)
    for t in range(T+1):
        Delta.requires_grad = True
        gamma = torch.rand(N, 1, 1, 1, 1).to(device)
        gamma = gamma / gamma.sum(0).item()
        theta = (gamma * Delta).sum(0) 
        mim_image = image + theta
        mim_image = torch.clamp(mim_image, min=minx, max=maxx)#需要吗
        mim_image = torch.clamp(mim_image, min=0, max=1)       
        y_hat = model(mim_image)[0].reshape(1,-1)
        l = range(N)
        j, k = 0, 0
        while j==k:
            j = choice(l)
            k = choice(l)

        delta1 = Delta[j,:,:]
        delta2 = Delta[k,:,:]
        loss = criterion(y_hat, y_ture) - beta * cosine_similarity(delta1, delta2)**2
        loss.backward()
        if t==0:
            Delta=Delta.grad.data.clone().detach()
            continue

        grad = Delta.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad + momentum * mu
        momentum = grad

        Delta = Delta + alpha * torch.sign(grad)
        Delta = Delta.clone().detach()
        delta_center = torch.clamp(Delta.sum(0)/N, min=-eps, max=eps)
        adv_image_center = torch.clamp(image + delta_center, min=0, max=1).detach()
    return adv_image_center
def mim_attack(ori_image, eps, iter_num, model, criterion, mu, target, device):
    # pdb.set_trace() 
    alpha = eps / iter_num
    image = ori_image.clone().detach().to(device)
    target = target.clone().detach().to(device)

    momentum = torch.zeros_like(image).detach().to(device)
    adv_image = image.clone().detach()
    # pdb.set_trace()
    for _ in range(iter_num):
        adv_image.requires_grad = True
        output = model(adv_image)[0].reshape(1,-1)
        loss = criterion(output,target)

        # model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)#!!
        grad = grad + momentum * mu
        momentum = grad

        adv_image.data = adv_image.data + alpha * momentum.sign()
        adv_image.grad.zero_()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image
