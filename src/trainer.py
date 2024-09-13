def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            texts, labels = batch['text'], batch['label']
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        valid_loss, valid_f1 = validate_model(model, criterion, valid_loader, device)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {valid_loss}, Validation F1 Score: {valid_f1}')

def validate_model(model, criterion, valid_loader, device):
    model.eval()
    valid_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            texts, labels = batch['text'], batch['label']
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return valid_loss / len(valid_loader), f1


