/*
    generate bird eye view of path
*/
const mineflayer = require('mineflayer')
const mineflayerViewer = require('prismarine-viewer').mineflayer
const {get_target, targets_village} = require('./targets.js')
const Vec3 = require('vec3').Vec3
const fs = require('fs')  // 添加 fs 模块


const bot = mineflayer.createBot({
  username: 'Bot',
  host: 'localhost',
  port: 8105
})
let location = "plains_village_2"
let nvtype = "ABCA"
let tp_target = get_target(location)


bot.once('spawn', () => {
  bot.chat(`/tp @p ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  // wait for 1 seconds to let the bot tp
  mineflayerViewer(bot, { port:3008, firstPerson: false, width: 640, height: 360 })
  setTimeout(() => {
    
    const fs = require('fs')
    const path = `/nfs-shared-2/liankewei-mus2/data/${nvtype}/${location}/5`
    const path2 = `/nfs-shared-2/liankewei-mus2/data/${nvtype}/${location}/15`
    const path3 = `/nfs-shared-2/liankewei-mus2/data/${nvtype}/${location}/30`


    const linePath1 = []
    const linePath2 = []
    const linePath3 = []
    // helper: 加载路径点
    function loadPathFromDir(dir, linePathArray) {
      const files = fs.readdirSync(dir)
      for (const file of files) {
        if (!file.endsWith(".json")) continue
        const data = fs.readFileSync(`${dir}/${file}`, 'utf8')
        const jsonArray = JSON.parse(data)
        for (const node of jsonArray) {
          linePathArray.push({ x: node.x, y: node.y + 0.5, z: node.z })
        }
      }
    }

    // 加载两个目录下的路径
    loadPathFromDir(path, linePath1)
    loadPathFromDir(path2, linePath2)
    loadPathFromDir(path3, linePath3)
    // 绘制
    bot.viewer.drawPoints('path1', linePath1, 0xffb051, 3)  // 橙色
    bot.viewer.drawPoints('path2', linePath2, 0x51b0ff, 3)  // 蓝色
    // bot.viewer.drawPoints('path3', linePath3, 0x92CFA3, 3)  
  
  }, 1000)
})
