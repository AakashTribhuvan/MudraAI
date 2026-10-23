import React from 'react'
import { createRoot } from 'react-dom/client'
import AvatarLab from './AvatarLab.jsx'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AvatarLab />
  </React.StrictMode>,
)
