import React, { useState } from 'react';
import { Container, Typography, TextField, Button, Paper, List, ListItem, ListItemText } from '@mui/material';

function CrewAIChatUI() {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: 'Welcome to ClerkAI! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages([...messages, { sender: 'user', text: input }]);
    // TODO: Integrate with backend API for CrewAI agent response
    setMessages(prev => [...prev, { sender: 'assistant', text: 'This is a placeholder response from CrewAI.' }]);
    setInput('');
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Typography variant="h4" align="center" gutterBottom>
        CrewAI Chat UI
      </Typography>
      <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
        <List>
          {messages.map((msg, idx) => (
            <ListItem key={idx}>
              <ListItemText primary={msg.text} secondary={msg.sender === 'user' ? 'You' : 'Assistant'} />
            </ListItem>
          ))}
        </List>
      </Paper>
      <TextField
        fullWidth
        label="Type your message..."
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' ? handleSend() : null}
      />
      <Button variant="contained" color="primary" fullWidth sx={{ mt: 2 }} onClick={handleSend}>
        Send
      </Button>
    </Container>
  );
}

export default CrewAIChatUI;
