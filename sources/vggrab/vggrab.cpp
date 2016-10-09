#include "grabber.hpp"

#define WINDOW_TITLE    "VGGRABER - v1.0.0"
#define BUFFER_SIZE     100
#define SYNC_FPS        15
#define SYNC_TOL        0.9
#define WAIT_DELAY      50
#define KEY_QUIT        'q'
#define KEY_RECORDING   ' '
#define KEY_SNAPSHOT    's'
#define KEY_VIEW_PREV   'c'
#define KEY_VIEW_NEXT   'v'
#define KEY_PLOT_SWITCH 'l'

using namespace seq2map;

int main(int argc, char* argv[])
{
    initLogFile("vggrab.log");

    // Buffer writers & reader
    SyncBuffer buffer(2, 1, BUFFER_SIZE, SYNC_FPS, SYNC_TOL);
    //StImageGrabber cam0(buffer), cam1(buffer);
    DummyImageGrabber cam0(buffer), cam1(buffer);
    BufferRecorder recorder(buffer, "seq");

    // UI stuff
    cv::Mat canvas;
    StereoImageRenderer imageRenderer(cam0, cam1);
    std::vector<BufferWriterStatsRenderer> statsRenderers;
    BufferUsageIndicator usageIndicator(buffer);
    BufferRecorderStatsRenderer recRenderer(recorder);

    statsRenderers.push_back(BufferWriterStatsRenderer(cam0, 1000 / WAIT_DELAY * 3));
    statsRenderers.push_back(BufferWriterStatsRenderer(cam1, 1000 / WAIT_DELAY * 3));
    statsRenderers[0].Rectangle = cv::Rect(32, 32, 320, 90);
    statsRenderers[1].Rectangle = cv::Rect(32, 138, 320, 90);

    size_t viewIdx = 0, numViews = StereoImageRenderer::ListedModes.size();
    imageRenderer.SetMode(StereoImageRenderer::ListedModes[viewIdx]);

    usageIndicator.Rectangle = cv::Rect(-192, -64, 160, 32);
    recRenderer.Origin = cv::Point(-400, 64);

    cv::namedWindow(WINDOW_TITLE, cv::WINDOW_AUTOSIZE);
 
    if (!cam0.Start() || !cam1.Start() || !recorder.Start()) return -1;

    for (int key = 0; key != KEY_QUIT; key = cv::waitKey(WAIT_DELAY))
    {
        switch (key) 
        {
        case KEY_VIEW_PREV:
            imageRenderer.SetMode(StereoImageRenderer::ListedModes[--viewIdx % numViews]);
            break;
        case KEY_VIEW_NEXT:
            imageRenderer.SetMode(StereoImageRenderer::ListedModes[++viewIdx % numViews]);
            break;
        case KEY_RECORDING:
            if (!recorder.IsRecording()) recorder.StartRecording();
            else recorder.StopRecording();
            break;
        case KEY_SNAPSHOT:
            recorder.Snapshot();
            break;
        case KEY_QUIT: //
            break;
        default:
            /*****/;
        }
        
        if (!imageRenderer.Draw(canvas))
        {
            E_ERROR << "error rendering camera views";
            continue;
        }

        BOOST_FOREACH(BufferWriterStatsRenderer& render, statsRenderers)
        {
            if (render.Draw(canvas)) continue;
            E_ERROR << "error rendering stats";
        }

        if (!usageIndicator.Draw(canvas)) E_ERROR << "error rendering buffer usage indicator";
        if (!recRenderer.Draw(canvas)) E_ERROR << "error rendering recorder stats";

        cv::imshow(WINDOW_TITLE, canvas);
    }

    cam0.Stop();
    cam1.Stop();
    recorder.Stop();

    return 0;
}
