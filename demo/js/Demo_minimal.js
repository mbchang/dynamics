/**
* The Matter.js demo page controller and example runner.
*
* NOTE: For the actual example code, refer to the source files in `/examples/`.
*
* @class Demo
*/

(function() {

    var _isBrowser = typeof window !== 'undefined' && window.location,
        _useInspector = _isBrowser && window.location.hash.indexOf('-inspect') !== -1,
        _isMobile = _isBrowser && /(ipad|iphone|ipod|android)/gi.test(navigator.userAgent),
        _isAutomatedTest = !_isBrowser || window._phantom;

    // var Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    var Matter = _isBrowser ? window.Matter : require('matter-js');

    var Demo = {};
    Matter.Demo = Demo;

    if (!_isBrowser) {
        var jsonfile = require('jsonfile')
        var CircularJSON = require('circular-json')
        var assert = require('assert')
        var utils = require('../../utils')
        var sleep = require('sleep')
        // var PImage = require('pureimage');
        require('./Examples')
        var env = process.argv.slice(2)[0]
        if (env == null)
            throw('Please provide an enviornment, e.g. node Demo.js hockey')
        module.exports = Demo;
        window = {};
    }

    // Matter aliases
    var Body = Matter.Body,
        Example = Matter.Example,
        Engine = Matter.Engine,
        World = Matter.World,
        Common = Matter.Common,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies,
        Events = Matter.Events,
        Runner = Matter.Runner,
        Render = Matter.Render;

    // Create the engine
    Demo.run = function(json_data) {

        //TODO: note that here you should load the demo engine with the json file

        // load the config file here.
        console.log(json_data)
        let data = json_data.trajectories
        let config = json_data.config
        console.log(config)

        // let data = json_data

        var demo = {}
        demo.engine = Engine.create()
        demo.runner = Engine.run(demo.engine)
        demo.container = document.getElementById('canvas-container');
        // demo.render = Render.create({element: demo.container, engine: demo.engine})
        // Render.run(demo.render)

        demo.offset = 5;  // world offset
        demo.config = {}
        demo.config.cx = 400;
        demo.config.cy = 300;
        demo.config.masses = [1, 5, 25]
        demo.config.mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}
        demo.config.sizes = [2/3, 1, 3/2]  // multiples
        demo.config.object_base_size = {'ball': 60, 'obstacle': 80, 'block': 20 }  // radius of ball, side of square obstacle, long side of block
        demo.config.objtypes = ['ball', 'obstacle', 'block']  // squares are obstacles
        demo.config.g = 0 // default? [0,1] Or should we make this a list? The index of the one hot. 0 is no, 1 is yes
        demo.config.f = 0 // default? [0,1]
        demo.config.p = 0 // default? [0,1,2]
        demo.config.max_velocity = 60

        demo.cx = demo.config.cx;
        demo.cy = demo.config.cy;
        demo.width = 2*demo.cx
        demo.height = 2*demo.cy

        demo.engine.world.bounds = { min: { x: 0, y: 0 },
                            max: { x: demo.width, y: demo.height }}

        var world_border = Composite.create({label:'Border'});

        Composite.add(world_border, [
            Bodies.rectangle(demo.cx, -demo.offset, demo.width + 2*demo.offset, 2*demo.offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(demo.cx, demo.height+demo.offset, demo.width + 2*demo.offset, 2*demo.offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(demo.width + demo.offset, demo.cy, 2*demo.offset, demo.height + 2*demo.offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(-demo.offset, demo.cy, 2*demo.offset, demo.height + 2*demo.offset, { isStatic: true, restitution: 1 })
        ]);

        World.add(demo.engine.world, world_border)  // its parent is a circular reference!

        demo.render = Render.create({element: demo.container, engine: demo.engine, 
                                    hasBounds: true, options:{height:demo.height, width:demo.width}})
        Render.run(demo.render)

        Events.on(demo.render, 'afterRender', function(e) {
            var img = demo.render.canvas.toDataURL("image"+e.timestamp+".png")
            var data = img.replace(/^data:image\/\w+;base64,/, "");
            // var buf = new Buffer(data, 'base64');
            // fs.writeFile('image.png', buf);
            // document.write('<img src="'+img+'"/>');  // how do I save this?
            // PImage.encodePNG(img, fs.createWriteStream('out.png'), function(err) {
            // console.log("wrote out the png file to out.png");
            // });
        });

        if (demo.render) {
            var renderOptions = demo.render.options;
            renderOptions.wireframes = false;
            renderOptions.hasBounds = false;
            renderOptions.showDebug = false;
            renderOptions.showBroadphase = false;
            renderOptions.showBounds = false;
            renderOptions.showVelocity = false;
            renderOptions.showCollisions = false;
            renderOptions.showAxes = false;
            renderOptions.showPositions = false;
            renderOptions.showAngleIndicator = false;
            renderOptions.showIds = false;
            renderOptions.showShadows = false;
            renderOptions.showVertexNumbers = false;
            renderOptions.showConvexHulls = false;
            renderOptions.showInternalEdges = false;
            renderOptions.showSeparations = false;
            renderOptions.background = '#fff';

            if (_isMobile) {
                renderOptions.showDebug = true;
            }
        }

        var mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}// TODO eventually call Example[config.env].mass_colors



        // Ok, now let's manually update
        Runner.stop(demo.runner)

        var trajectories = data[8]  // extra 0 for batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length
        config.trajectories = trajectories

        Example[config.env](demo, config)  // here you have to assign balls initial positions according to the initial timestep of trajectories.

        // here you have to 
        console.log(config)

        console.log(num_steps)
        if (config.env=='tower') {
            var i = 2  // if I set i to < 2 then I get very weird behavior
        } else {
            var i = 0
        }

        function f() {
            console.log( 'i', i );
            var entities = Composite.allBodies(demo.engine.world)
                .filter(function(elem) {
                            return elem.label === 'Entity';
                        })
            var entity_ids = entities.map(function(elem) {
                                return elem.id});

            for (id = 0; id < entity_ids.length; id++) { //id = 0 corresponds to world!
                var body = Composite.get(demo.engine.world, entity_ids[id], 'body')
                // set the position here
                // body.render.fillStyle = mass_colors[trajectories[id][i].mass]//'#4ECDC4'
                if (i < config.num_past) {
                    body.render.strokeStyle = '#FFA500'// orange #551A8B is purple
                } else {
                    body.render.strokeStyle = '#551A8B'// orange #551A8B is purple
                }
                body.render.lineWidth = 5

                console.log('set position')
                // let eps = 0.0001  // to prevent bounce-back


                Body.setPosition(body, trajectories[id][i].position)


                // if (id == 0) {
                //     Body.setPosition(body, trajectories[id][i].position)
                // }

                // Body.setVelocity(body, trajectories[id][i].velocity)

                // console.log(trajectories[id][i].position)
                // console.log(trajectories[id][i].velocity)

                Body.setAngle(body, trajectories[id][i].angle)
                if (trajectories[id][i].mass == 1) {
                    if (id==0) {
                        console.log(id)
                        console.log('traj vel', trajectories[id][i].velocity)
                        console.log('bod vel', body.velocity)
                        console.log('traj pos', trajectories[id][i].position)
                        console.log('bod pos', body.position)
                        console.log('traj ang', trajectories[id][i].angle)
                        console.log('bod ang', body.angle)
                        console.log('traj angvel', trajectories[id][i].angularVelocity)
                        console.log('bod angvel', body.angularVelocity)
                    }
                }
            }

            Runner.tick(demo.runner, demo.engine);
            i++;
            if( i < num_steps ){
                setTimeout( f, 50 );
            }
        }
        f();


    }

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        window.loadFile = function loadFile(file){
            var fr = new FileReader();
            fr.onload = function(){
                Demo.run(window.CircularJSON.parse(fr.result))
            }
            fr.readAsText(file)
        }
    }
})();
