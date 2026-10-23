const { app, BrowserWindow, Menu, session, shell } = require('electron')
const path = require('node:path')

// Remove the File/Edit/View application menus while keeping the native title bar.
Menu.setApplicationMenu(null)

function createWindow() {
  const window = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 900,
    minHeight: 650,
    autoHideMenuBar: true,
    backgroundColor: '#FAF7F0',
    title: 'SignLine',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })

  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://') || url.startsWith('http://')) shell.openExternal(url)
    return { action: 'deny' }
  })

  const devUrl = process.env.VITE_DEV_SERVER_URL
  if (devUrl) window.loadURL(devUrl)
  else window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  window.maximize()
}

app.whenReady().then(() => {
  // The app asks only for camera access; audio capture is not used.
  session.defaultSession.setPermissionCheckHandler((_webContents, permission, _origin, details) => (
    permission === 'media' && details?.mediaTypes?.includes('video') && !details.mediaTypes.includes('audio')
  ))
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback, details) => {
    const cameraOnly = details?.mediaTypes?.includes('video') && !details.mediaTypes.includes('audio')
    callback(permission === 'media' && cameraOnly)
  })

  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
