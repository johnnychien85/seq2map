#include <boost/assign/list_of.hpp>
#include "grabber.hpp"
#include "uirenderer.hpp"

#define KEY_QUIT        'q'
#define KEY_RECORDING   ' '
#define KEY_SNAPSHOT    'e'
#define KEY_UP          'w'
#define KEY_DOWN        's'
#define KEY_LEFT        'a'
#define KEY_RIGHT       'd'
#define KEY_PLOT_SWITCH 'v'

const cv::Point UIRenderer::Origin = cv::Point(0, 0);
size_t UIRenderer::s_newId = 0;

const String MainUI::s_caption  = "VGGRAB - Soft-sync Multi-camera Frame Grabber 1.0";
const int    MainUI::s_waitKeyTimeout = 50;

const size_t CascadedMenu::s_itemHeight = 22;
const size_t CascadedMenu::s_vspace = 2;
const size_t CascadedMenu::s_hspace = 32;

cv::Mat bgr2gray(const cv::Mat& im)
{
    cv::Mat gray = im;
    if (im.channels() == 3) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

    return gray;
}

bool UIRenderer::Render(cv::Mat& canvas, const cv::Point& origin)
{
    if (!Visible) return true;

    if (!Draw(canvas, origin))
    {
        return false;
    }

    bool doBlending = AlphaBlending < 1.0f;
    cv::Mat overlay = doBlending ? cv::Mat::zeros(canvas.size(), canvas.type()) : canvas;

#ifdef DEBUG
    cv::rectangle(overlay, cv::Rect(origin.x + Rectangle.x, origin.y + Rectangle.y, Rectangle.width, Rectangle.height), cv::Scalar(0, 255, 0), 1);
#endif

    BOOST_FOREACH (Ptr child, m_children)
    {

        cv::Point absOrigin = origin + Rectangle.tl();
        cv::Point relOrigin = child->Rectangle.tl();

        if (!child->AlignToLeft) absOrigin.x += Rectangle.width  - child->Rectangle.width  - 2 * relOrigin.x;
        if (!child->AlignToTop ) absOrigin.y += Rectangle.height - child->Rectangle.height - 2 * relOrigin.y;

        if (!child->Render(overlay, absOrigin))
        {
            return false;
        }
    }

    if (doBlending) canvas = canvas + overlay * AlphaBlending;

    return true;
}

MainUI::MainUI(const ImageGrabber::Ptrs& grabbers, BufferRecorder& recorder, const SyncBuffer::Ptr& buffer)
: m_grabbers(grabbers), m_recorder(recorder), m_buffer(buffer), m_flashing(0),
  m_vmF3idx(INVALID_INDEX), m_vmF2idx(INVALID_INDEX),
  m_vmS3idx(INVALID_INDEX), m_vmS2idx(INVALID_INDEX), m_vmS1idx(INVALID_INDEX)
{
    cv::namedWindow(s_caption, cv::WINDOW_AUTOSIZE);
    CascadedMenu::Expandable::Ptrs itemsNeedGrabberList;

    m_menu = CascadedMenu::Create();

    CascadedMenu::Ptr mainMenu = m_menu->CreateExpandableItem("D")->GetSubmenu();
    CascadedMenu::Expandable::Ptr vm = mainMenu->CreateExpandableItem("Rendering");
    CascadedMenu::Expandable::Ptr sm = mainMenu->CreateExpandableItem("Speedometre");

    if (grabbers.size() > 2)
    {
        CascadedMenu::Expandable::Ptr f3 = vm->GetSubmenu()->CreateExpandableItem("3-FUSION", 3);
        CascadedMenu::Expandable::Ptr s3 = vm->GetSubmenu()->CreateExpandableItem("3-SIDE",   3);
        m_vmF3idx = f3->GetIndex();
        m_vmS3idx = s3->GetIndex();
        itemsNeedGrabberList.push_back(f3);
        itemsNeedGrabberList.push_back(s3);

        if (!m_activeViewMenu) m_activeViewMenu = f3->GetSubmenu();
    }
    if (grabbers.size() > 1)
    {
        CascadedMenu::Expandable::Ptr f2 = vm->GetSubmenu()->CreateExpandableItem("2-FUSION", 2);
        CascadedMenu::Expandable::Ptr s2 = vm->GetSubmenu()->CreateExpandableItem("2-SIDE",   2);
        m_vmF2idx = f2->GetIndex();
        m_vmS2idx = s2->GetIndex();
        itemsNeedGrabberList.push_back(f2);
        itemsNeedGrabberList.push_back(s2);

        if (!m_activeViewMenu) m_activeViewMenu = f2->GetSubmenu();
    }
    if (grabbers.size() > 0)
    {
        CascadedMenu::Expandable::Ptr s1 = vm->GetSubmenu()->CreateExpandableItem("MONOVIEW", 1);
        m_vmS1idx = s1->GetIndex();
        itemsNeedGrabberList.push_back(s1);

        if (!m_activeViewMenu) m_activeViewMenu = s1->GetSubmenu();
    }

    itemsNeedGrabberList.push_back(sm);

    BOOST_FOREACH(const CascadedMenu::Expandable::Ptr& expandable, itemsNeedGrabberList)
    {
        BOOST_FOREACH(const ImageGrabber::Ptr& grabber, grabbers)
        {
            expandable->GetSubmenu()->CreateTogglableItem(grabber->ToString());
        }
    }

    m_menu->OnResized();
    m_menu->AlignToTop = false;
    m_menu->Rectangle.y = 8;

    vm->GetSubmenu()->SetListener(this);
    sm->GetSubmenu()->SetListener(this);

    m_vmMenuId = vm->GetSubmenu()->GetId();
    m_smMenuId = sm->GetSubmenu()->GetId();

    OnItemChanged(*vm->GetSubmenu(), 0);

    m_menu->NextItem();

    cv::Point pt(16, 16);
    BOOST_FOREACH(const ImageGrabber::Ptr& grabber, grabbers)
    {
        UIRenderer::Ptr plot = UIRenderer::Ptr(new BufferWriterStatsRenderer(grabber));
        plot->Rectangle.x = pt.x;
        plot->Rectangle.y = pt.y;

        pt.y += plot->Rectangle.height + 12;

        AddChild(plot);
    }

    m_bufUsageIndicator = UIRenderer::Ptr(new BufferUsageIndicator(buffer));
    m_bufUsageIndicator->AlignToLeft = false;
    m_bufUsageIndicator->AlignToTop = false;
    m_bufUsageIndicator->Rectangle.width  = 120;
    m_bufUsageIndicator->Rectangle.height = 50;
    m_bufUsageIndicator->Rectangle.x = 16;
    m_bufUsageIndicator->Rectangle.y = 16;
    m_bufUsageIndicator->OnResized();

    m_recorderIndicator = UIRenderer::Ptr(new RecorderStatsRenderer(recorder));
    m_recorderIndicator->AlignToLeft = false;
    m_recorderIndicator->Rectangle.width = 240;
    m_recorderIndicator->Rectangle.height = 82;
    m_recorderIndicator->Rectangle.x = 16;
    m_recorderIndicator->Rectangle.y = 16;
    m_recorderIndicator->OnResized();

    AddChild(m_menu);
    AddChild(m_bufUsageIndicator);
    AddChild(m_recorderIndicator);

    AlphaBlending = 0.9;
}

void MainUI::Loop()
{
    for (int key = 0; key != KEY_QUIT; key = cv::waitKey(s_waitKeyTimeout))
    {
        cv::Mat canvas;
        bool pressed = key > 0;

        if (pressed) KeyPressed(key);
        if (Render(canvas) && !canvas.empty()) cv::imshow(s_caption, canvas);
    }
}

void MainUI::KeyPressed(int key)
{
    switch (key)
    {
    case KEY_RECORDING:
        if (!m_recorder.IsRecording()) m_recorder.StartRecording();
        else m_recorder.StopRecording();
        break;
    case KEY_SNAPSHOT:
        m_recorder.Snapshot();
        m_flashing = 255;
        break;
    case KEY_UP:
        m_menu->PrevItem();
        break;
    case KEY_DOWN:
        m_menu->NextItem();
        break;
    case KEY_LEFT:
        m_menu->PrevLevel();
        break;
    case KEY_RIGHT:
        m_menu->NextLevel();
        break;
    case KEY_PLOT_SWITCH:
        break;
    case KEY_QUIT:
    default:
        ;
    }
}

void MainUI::OnItemChanged(const CascadedMenu& source, size_t itemIdx)
{
    if (source.GetId() == m_vmMenuId)
    {
        const CascadedMenu::Expandable::Ptr expandable = boost::dynamic_pointer_cast<CascadedMenu::Expandable>(source.GetItem(itemIdx));
        if (!expandable) return;

        m_activeViewMenu = expandable->GetSubmenu();
        m_mixchan = itemIdx == m_vmF2idx || itemIdx == m_vmF3idx;
    }
}

bool MainUI::Draw(cv::Mat& canvas, const cv::Point& origin)
{
    if (!m_activeViewMenu) return false;

    std::vector<size_t> idx = m_activeViewMenu->GetToggled();
    size_t n = idx.size();

    if (m_mixchan)
    {
        std::vector<cv::Mat> bgr(3);
        assert(n == 2 || n == 3);

        if (n == 2)
        {
            bgr[0] = bgr2gray(m_grabbers[idx[0]]->GetImage());
            bgr[2] = bgr2gray(m_grabbers[idx[1]]->GetImage());

            if (bgr[0].empty() || bgr[2].empty()) return false;

            bgr[1] = bgr[0] * 0.5f + bgr[2] * 0.5;
        }
        else
        {
            for (size_t i = 0; i < 3; i++)
            {
                bgr[i] = bgr2gray(m_grabbers[idx[i]]->GetImage());

                if (bgr[i].empty()) return false;
            }
        }

        cv::merge(bgr, canvas);
    }
    else
    {
        assert(n > 0);

        std::vector<cv::Mat> im(n);
        cv::Size canvasSize;

        for (size_t i = 0; i < n; i++)
        {
            im[i] = m_grabbers[idx[i]]->GetImage();

            if (im[i].empty()) return false;

            canvasSize.width += im[i].cols;
            canvasSize.height = std::max(canvasSize.height, im[i].rows);

            if (im[i].channels() == 1) cv::cvtColor(im[i], im[i], cv::COLOR_GRAY2BGR);
        }

        canvas = cv::Mat::zeros(canvasSize, CV_8UC3);
        cv::Rect roi;

        for (size_t i = 0; i < n; i++)
        {
            roi.width  = im[i].cols;
            roi.height = im[i].rows;
            im[i].copyTo(canvas(roi));

            roi.x += roi.width;
        }
    }

    Rectangle.x = 0;
    Rectangle.y = 0;
    Rectangle.width = canvas.cols;
    Rectangle.height = canvas.rows;

    if (!canvas.empty())
    {
        if (canvas.channels() == 1)
        {
            cv::cvtColor(canvas, canvas, CV_GRAY2BGR);
        }

        if (m_flashing > 10)
        {
            canvas += cv::Scalar(m_flashing, m_flashing, m_flashing);
            m_flashing /= 2;
        }
    }

    return !canvas.empty();
}

CascadedMenu::Togglable::Ptr CascadedMenu::CreateTogglableItem(const String& itemName)
{
    Togglable::Ptr newItem = Togglable::Ptr(new Togglable(itemName, false));
    AddItem(static_cast<MenuItem::Ptr>(newItem));

    if (m_toggles == 0 || m_toggled.size() < m_toggles)
    {
        UpdateToggle(newItem, true);
    }

    return newItem;
}

CascadedMenu::Expandable::Ptr CascadedMenu::CreateExpandableItem(const String& caption, size_t toggles)
{
    Expandable::Ptr newItem = Expandable::Ptr(new Expandable(caption, toggles));
    AddItem(static_cast<MenuItem::Ptr>(newItem));

    CascadedMenu::Ptr submenu = newItem->GetSubmenu();
    submenu->Rectangle.x += newItem->Rectangle.width + newItem->Rectangle.width * 0.3;
    submenu->Rectangle.y =  newItem->Rectangle.y;
    submenu->Visible = false;

    AddChild(submenu);

    return newItem;
}

void CascadedMenu::AddItem(const CascadedMenu::MenuItem::Ptr& item)
{
    //MenuItem::Ptr lastItem = m_items.size() > 0 ? m_items.back() : MenuItem::Ptr();
    //item->Rectangle.y += lastItem ? (lastItem->Rectangle.br().y + lastItem->Paddings * 2) : 0;
    //item->Rectangle.width = 240;
    item->SetIndex(m_items.size());
    item->SetBackColour(m_darkColour1);
    item->SetForeColour(m_lightColour);

    m_items.push_back(item);

    AddChild(item);
}

void CascadedMenu::NextItem(int inc)
{
    if (m_items.empty()) return;

    MenuItem  ::Ptr highlighted = m_items[m_highlight];
    Togglable ::Ptr togglable   = boost::dynamic_pointer_cast<Togglable> (highlighted);
    Expandable::Ptr expandable  = boost::dynamic_pointer_cast<Expandable>(highlighted);

    // forward the item browsing to next level if the current selection
    // is expanded
    if (expandable && expandable->IsExpanded())
    {
        expandable->GetSubmenu()->NextItem(inc);
    }
    else
    {
        m_highlight = (m_highlight + inc) % m_items.size();

        highlighted->Highlighted = false; //togglable && togglable->IsToggled();
        m_items[m_highlight]->Highlighted = true;

        if (m_listener != NULL)
        {
            m_listener->OnItemChanged(*this, m_highlight);
        }
    }
}

bool CascadedMenu::NextLevel()
{
    if (m_items.empty()) return false;

    MenuItem  ::Ptr highlighted = m_items[m_highlight];
    Togglable ::Ptr togglable   = boost::dynamic_pointer_cast<Togglable> (highlighted);
    Expandable::Ptr expandable  = boost::dynamic_pointer_cast<Expandable>(highlighted);

    if (expandable)
    {
        if (expandable->IsExpanded())
        {
            expandable->GetSubmenu()->NextLevel();
        }
        else
        {
            if (expandable->CanExpand())
            {
                expandable->SetExpanded(true);
                expandable->GetSubmenu()->Visible = true;
                expandable->GetSubmenu()->NextItem(0);
            }

            return true;
        }
    }
    else if (togglable)
    {
        UpdateToggle(togglable, !togglable->IsToggled());
    }

    return false;
}

bool CascadedMenu::PrevLevel()
{
    if (m_items.empty()) return false;

    MenuItem  ::Ptr highlighted = m_items[m_highlight];
    Expandable::Ptr expandable  = boost::dynamic_pointer_cast<Expandable>(highlighted);

    if (expandable && expandable->IsExpanded())
    {
        if (expandable->GetSubmenu()->PrevLevel())
        {
            expandable->SetExpanded(false);
            expandable->GetSubmenu()->Visible = false;
        }

        return false;
    }

    return true;
}

void CascadedMenu::OnResized()
{
    size_t n = m_items.size();
    size_t itemWidth = 0;
    const size_t itemHeight = s_itemHeight;

    // determinie the maximum item size
    for (size_t i = 0; i < n; i++)
    {
        cv::Size itemSize = m_items[i]->GetTextSize();
        itemSize.width  += 12;
        itemWidth = itemWidth < itemSize.width ? itemSize.width : itemWidth;
    }

    Rectangle.width  = (itemWidth  + s_hspace);
    Rectangle.height = (itemHeight + s_vspace) * n - s_vspace;

    size_t itemHalfHeight = itemHeight / 2;

    // apply the optimal size to the menu items
    for (size_t i = 0; i < n; i++)
    {
        cv::Rect& rect = m_items[i]->Rectangle;
        rect.x = 0;
        rect.y = (s_itemHeight + s_vspace) * i;
        rect.width  = itemWidth;
        rect.height = itemHeight;
    }

    // recursively recalculate all the submenus
    for (size_t i = 0; i < n; i++)
    {
        Expandable::Ptr expandable = boost::dynamic_pointer_cast<Expandable>(m_items[i]);
        if (!expandable) continue;

        CascadedMenu::Ptr submenu = expandable->GetSubmenu();

        submenu->OnResized();
        submenu->Rectangle.x = itemWidth + s_hspace;
        submenu->Rectangle.y = expandable->Rectangle.y + itemHalfHeight - submenu->Rectangle.height / 2;

        int dy = Rectangle.br().y - submenu->Rectangle.br().y;

        if (dy < 0)
        {
            submenu->Rectangle.y += dy;
        }
    }
}

bool CascadedMenu::Draw(cv::Mat& canvas, const cv::Point& origin)
{
    if (m_items.empty()) return true;

    MenuItem::Ptr highlighted = m_items[m_highlight];
    Expandable::Ptr expandable = boost::dynamic_pointer_cast<Expandable>(highlighted);

    if (expandable)
    {
        CascadedMenu::Ptr submenu = expandable->GetSubmenu();

        if (expandable->IsExpanded())
        {
            const cv::Rect& itemRect = expandable->Rectangle;
            const cv::Rect& menuRect = submenu->Rectangle;

            cv::Point start;
            cv::Point mid0;

            start.x = origin.x + Rectangle.x + itemRect.x + itemRect.width;
            start.y = origin.y + Rectangle.y + itemRect.y + itemRect.height / 2;

            mid0.x = start.x + (menuRect.x - itemRect.br().x) / 2;
            mid0.y = start.y;

            cv::line(canvas, start, mid0, m_lightColour, 2);

            for (size_t i = 0; i < submenu->m_items.size(); i++)
            {
                MenuItem::Ptr subitem = submenu->m_items[i];
                bool highlighted = subitem->GetIndex() == submenu->m_highlight;
                int thickness = highlighted ? 2 : 1;
                cv::Scalar colour = highlighted ? m_lightColour : m_lightColour * 0.5f;

                cv::Point mid1;
                cv::Point term;

                mid1.x = mid0.x;
                mid1.y = origin.y + Rectangle.y + menuRect.y + subitem->Rectangle.y + subitem->Rectangle.height / 2;
                term.x = origin.x + Rectangle.x + menuRect.x + subitem->Rectangle.x;
                term.y = mid1.y;

                cv::line(canvas, mid0, mid1, colour, thickness);
                cv::line(canvas, mid1, term, colour, thickness);
            }
        }
        else if (expandable->CanExpand())
        {
            cv::Rect rect = highlighted->Rectangle;
            rect.x += origin.x + Rectangle.x + rect.width + 2;
            rect.y += origin.y + Rectangle.y + 4;
            rect.height -= 8;
            rect.width = 4;
            DrawArrow(canvas, rect, expandable->GetForeColour());
        }
    }

    return true;
}

void CascadedMenu::UpdateToggle(Togglable::Ptr togglable, bool toggled)
{
    size_t idx = togglable->GetIndex();

    if (toggled)
    {
        togglable->SetForeColour(m_darkColour2);
        togglable->SetBackColour(m_darkColour1);

        m_toggled.push_back(idx);

        if (m_toggles > 0 && m_toggled.size() > m_toggles)
        {
            Togglable::Ptr firstToggle = boost::dynamic_pointer_cast<Togglable>(m_items[m_toggled[0]]);
            assert(firstToggle);

            UpdateToggle(firstToggle, false);
        }
    }
    else
    {
        if (m_toggled.size() <= m_toggles)
        {
            return;
        }

        togglable->SetForeColour(m_lightColour);
        togglable->SetBackColour(m_darkColour1);

        for (std::vector<size_t>::iterator x = m_toggled.begin();
             x != m_toggled.end(); x++)
        {
             if (*x == idx)
             {
                m_toggled.erase(x);
                break;
             }
        }
    }

    togglable->SetToggled(toggled);
}

void CascadedMenu::DrawArrow(cv::Mat& canvas, const cv::Rect& rect, const cv::Scalar& colour)
{
    typedef std::vector<cv::Point> Points;
    typedef std::vector<Points> Polygons;

    Points pts;
    Polygons polygons;

    pts.push_back(cv::Point(rect.x, rect.y));
    pts.push_back(cv::Point(rect.x + rect.width, rect.y + rect.height / 2));
    pts.push_back(cv::Point(rect.x, rect.y + rect.height));

    polygons.push_back(pts);

    cv::fillPoly(canvas, polygons, colour);
}

BufferWriterStatsRenderer::BufferWriterStatsRenderer(const BufferWriter::Ptr& writer, size_t numRecords)
: m_writer(writer), m_records(BpsRecords(numRecords)), m_currentIdx(0), m_bestIdx(0)
{
    Rectangle.width  = 320;
    Rectangle.height = 120;

    AddChild(m_grabberLabel = Label::Create(m_writer->ToString()));
    AddChild(m_bpsLabel = Label::Create("bps"));
    AddChild(m_seqLabel = Label::Create("seq"));
    AddChild(m_fpsLabel = Label::Create("fps"));
    AddChild(m_delayLabel = Label::Create("-"));

    size_t textHeight = (size_t) m_seqLabel->Rectangle.height;
    size_t padding = 2;

    m_plotRect.x = padding;
    m_plotRect.y = padding;
    m_plotRect.width  = Rectangle.width - 2 * padding;
    m_plotRect.height = Rectangle.height - 2 * padding - textHeight;

    m_grabberLabel->HorizontalAlign = Label::Left;
    m_grabberLabel->Highlighted     = true;

    m_bpsLabel->HorizontalAlign = Label::Left;
    m_bpsLabel->Highlighted     = true;

    m_seqLabel->BorderWidth      = 0;
    m_seqLabel->Rectangle.width  = Rectangle.width * 0.4f;
    m_seqLabel->Rectangle.height = textHeight;
    m_seqLabel->Rectangle.x      = 0;
    m_seqLabel->Rectangle.y      = m_plotRect.br().y + padding;
    m_seqLabel->HorizontalAlign  = Label::Left;

    m_fpsLabel->Rectangle.width  = Rectangle.width * 0.3f;
    m_fpsLabel->Rectangle.height = m_seqLabel->Rectangle.height;
    m_fpsLabel->Rectangle.x      = m_seqLabel->Rectangle.br().x;
    m_fpsLabel->Rectangle.y      = m_seqLabel->Rectangle.y;
    m_fpsLabel->BorderWidth      = 0;
    m_fpsLabel->HorizontalAlign  = Label::Left;

    m_delayLabel->Rectangle.width  = Rectangle.width * 0.3f;
    m_delayLabel->Rectangle.height = m_fpsLabel->Rectangle.height;
    m_delayLabel->Rectangle.x      = m_fpsLabel->Rectangle.br().x;
    m_delayLabel->Rectangle.y      = m_fpsLabel->Rectangle.y;
    m_delayLabel->BorderWidth      = 0;
    m_delayLabel->HorizontalAlign  = Label::Left;
}

bool BufferWriterStatsRenderer::Draw(cv::Mat& canvas, const cv::Point& origin)
{
    Speedometre metre = m_writer->GetMetre();

    m_records[m_currentIdx] = metre.GetSpeed();
    m_bestIdx = m_records[m_currentIdx] > m_records[m_bestIdx] ? m_currentIdx : m_bestIdx;

    std::stringstream seq, fps, delay;

    seq   << "seq=" << std::setfill('0') << std::setw(8) << m_writer->GetSeq();
    fps   << std::fixed << std::setprecision(2) << std::setfill('0') << std::setw(2) << "fps="   << metre.GetFrequency();
    delay << std::fixed << std::setprecision(2) << std::setfill('0') << std::setw(2) << "delay=" << m_writer->GetDelta() << "ms";

    m_seqLabel->Caption = seq.str();
    m_fpsLabel->Caption = fps.str();
    m_delayLabel->Caption = delay.str();

    cv::Rect rect = Rectangle;
    cv::Rect plot = m_plotRect;

    rect.x += origin.x;
    rect.y += origin.y;

    plot.x += rect.x;
    plot.y += rect.y;

    cv::rectangle(canvas, rect, m_bpsLabel->GetForeColour());
    cv::rectangle(canvas, plot, m_bpsLabel->GetForeColour());

    cv::Point pt0, pt1(0, 0);

    for (size_t i = 0; i < m_records.size(); i++)
    {
        size_t idx = (m_currentIdx - i) % m_records.size();

        pt0 = pt1;
        pt1.x = plot.x + (plot.width  - 1) * (1.0f - (double)i / (double)m_records.size());
        pt1.y = plot.y + (plot.height - 1) * (1.0f - (m_records[m_bestIdx] > 0 ? (0.8f * m_records[idx] / m_records[m_bestIdx]) : 0.0f));

        if (i == 0)
        {
            std::stringstream ss;
            ss << (int) floor((metre.GetSpeed()) / 1024 / 1024) << "MB/s";

            int halfHeight = m_bpsLabel->Rectangle.height / 2;

            m_bpsLabel->Caption = ss.str();
            m_bpsLabel->Rectangle.x = pt1.x + 2 - origin.x - Rectangle.x;
            m_bpsLabel->Rectangle.y = pt1.y + 2 - origin.y - Rectangle.y - halfHeight;
        }
        else
        {
            line(canvas, pt0, pt1, m_bpsLabel->GetForeColour(), 2);
        }
    }

    m_currentIdx = (m_currentIdx + 1) % m_records.size();

    return true;
}

BufferUsageIndicator::BufferUsageIndicator(const SyncBuffer::Ptr& buffer)
: m_buffer(buffer)
{
    AddChild(m_usageLabel = Label::Create("USAGE"));

    m_barsRect.x = 0;
    m_barsRect.y = 0;

    m_usageLabel->Rectangle.x = 0;
    m_usageLabel->Rectangle.y = 0;
    m_usageLabel->HorizontalAlign = Label::Left;
    m_usageLabel->BorderWidth = 0;
}

void BufferUsageIndicator::OnResized()
{
    const size_t spacing = 2;

    m_barsRect.width  = Rectangle.width;
    m_barsRect.height = Rectangle.height - m_usageLabel->Rectangle.height - spacing;

    m_usageLabel->Rectangle.width  = Rectangle.width;
    m_usageLabel->Rectangle.y = m_barsRect.br().y + spacing;
}

bool BufferUsageIndicator::Draw(cv::Mat& canvas, const cv::Point& origin)
{
    const size_t n = 10;
    const double spacing = 2;
    double k = m_buffer->GetUsage();
    double freePercent = (1.0f - k) * 100.0f;

    std::stringstream ss;
    ss << "BUFFER: " << std::setprecision(0) << std::fixed << freePercent << "% FREE";
    m_usageLabel->Caption = ss.str();

    cv::Rect plot = m_barsRect;
    plot.x += origin.x + Rectangle.x;
    plot.y += origin.y + Rectangle.y;

    cv::Scalar colour1((1.0f - k) * 255.0f, (1.0f - 0.5f * k) * 255.0f, 255.0f);
    cv::Scalar colour2(255.0f, 255.0f, 255.0f);

    cv::rectangle(canvas, plot, colour2, 1);

    plot.x += spacing;
    plot.y += spacing;
    plot.width  -= 2 * spacing;
    plot.height -= 2 * spacing;

    double w = (double) (plot.width + spacing) / n;
    size_t p = (size_t) (n * (1.0f - k));

    for (size_t i = 0; i < n; i++)
    {
        cv::Rect blk;
        blk.width  = w - spacing;
        blk.height = plot.height;
        blk.x      = plot.x + i * w;
        blk.y      = plot.y;

        if (i <= p) rectangle(canvas, blk, colour1, -1);
        else        rectangle(canvas, blk, colour2, +1);
    }

    return true;
}

RecorderStatsRenderer::RecorderStatsRenderer(BufferRecorder& rec)
: m_rec(rec), m_seq(0)
{

    AddChild(m_recordingLabel = Label::Create("RECORDING"));
    AddChild(m_miscsLabel     = Label::Create("..."));

    cv::Scalar red(0, 0, 255);

    m_recordingLabel->SetForeColour(red);
    m_recordingLabel->FontScale = 0.8;
    m_recordingLabel->FontWeight = 2;
    m_recordingLabel->BorderWidth = 5;

    m_miscsLabel->Rectangle.height = 27;
    m_miscsLabel->HorizontalAlign = Label::Right;
    m_miscsLabel->BorderWidth = 0;
}

void RecorderStatsRenderer::OnResized()
{
    m_recordingLabel->Rectangle.width  = Rectangle.width;
    m_recordingLabel->Rectangle.height = Rectangle.height * 0.8;

    m_miscsLabel->Rectangle.width = Rectangle.width;
    m_miscsLabel->Rectangle.y = m_recordingLabel->Rectangle.br().y;
}

bool RecorderStatsRenderer::Draw(cv::Mat& canvas, const cv::Point& origin)
{
    if (!m_rec.IsRecording())
    {
        m_recordingLabel->Visible = m_miscsLabel->Visible = false;
        return true;
    }

    m_recordingLabel->Highlighted = m_seq < 7;
    m_recordingLabel->Visible = true;

    m_miscsLabel->Rectangle.y = m_recordingLabel->Rectangle.br().y + 1;
    m_miscsLabel->Visible = true;

    std::stringstream ss;
    ss << m_rec.GetDropped() << " DROPPED / " << m_rec.GetWritten() << " WRITTEN";
    m_miscsLabel->Caption = ss.str();

    m_seq = (m_seq + 1) % 10;

    return true;
}

void Label::MakeDisabledColour(const cv::Scalar& enabled, cv::Scalar& disabled)
{
    disabled = enabled * 0.5f;
}

Label::Label(const String& caption)
: m_backColour    (cv::Scalar(  0,  0,  0)),
  m_foreColour    (cv::Scalar(255,255,255)),
  Caption         (caption),
  BorderWidth     (1),
  FontFace        (cv::FONT_HERSHEY_COMPLEX_SMALL),
  FontScale       (0.5f),
  FontWeight      (1),
  Highlighted     (false),
  Filled          (false),
  Paddings        (2),
  VerticalAlign   (Middle),
  HorizontalAlign (Centre)
{
    SetBackColour(m_backColour);
    SetForeColour(m_foreColour);
}

cv::Size Label::GetTextSize() const
{
    int baseline = 0;
    return cv::getTextSize(Caption, FontFace, FontScale, FontWeight, &baseline);
}

bool Label::Draw(cv::Mat& canvas, const cv::Point& origin)
{
    cv::Rect outer = Rectangle;
    outer.x += origin.x;
    outer.y += origin.y;

    cv::Rect inner = outer;

    inner.x += Paddings;
    inner.y += Paddings;
    inner.width  -= 2 * Paddings;
    inner.height -= 2 * Paddings;

    // --------------------------------------------------------------------
    // Enabled   Hightlighted   Foreground Colour      Background Colour
    // --------------------------------------------------------------------
    //                          m_disabledForeColour   m_disabledBackColour
    //           v              m_disabledBackColour   m_disabledForeColour
    // v                        m_backColour           m_foreColour
    // v         v              m_foreColour           m_backColour
    // --------------------------------------------------------------------

    cv::Scalar& foreColour = Enabled ? (Highlighted ? m_backColour : m_foreColour) : (Highlighted ? m_disabledBackColour : m_disabledForeColour);
    cv::Scalar& backColour = Enabled ? (Highlighted ? m_foreColour : m_backColour) : (Highlighted ? m_disabledForeColour : m_disabledBackColour);

    if (BorderWidth > 0) // draw the bounding box
    {
        cv::rectangle(canvas, outer, foreColour, static_cast<int>(BorderWidth));
    }

    if (Filled || Highlighted) // draw background
    {
        cv::rectangle(canvas, inner, backColour, cv::FILLED);
    }

    int baseline = 0;
    cv::Size  textSize = cv::getTextSize(Caption, FontFace, FontScale, FontWeight, &baseline);
    cv::Point textOrigin = inner.tl();

    textOrigin.y += baseline + FontWeight;

    switch (HorizontalAlign)
    {
    case Centre: textOrigin.x += (inner.width - textSize.width) * 0.5f; break;
    case Right:  textOrigin.x += (inner.width - textSize.width);        break;
    }

    switch (VerticalAlign)
    {
    case Middle: textOrigin.y += (inner.height - textSize.height) * 0.5f; break;
    case Bottom: textOrigin.y += (inner.height - textSize.height);        break;
    }

    cv::putText(canvas, Caption, textOrigin, FontFace, FontScale, foreColour, FontWeight);

    return true;
}
