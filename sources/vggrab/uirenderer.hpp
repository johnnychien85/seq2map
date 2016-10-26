#ifndef UIRENDERER_HPP
#define UIRENDERER_HPP
#include "grabber.hpp"
#include "recorder.hpp"

class UIRenderer
{
public:
    typedef boost::shared_ptr<UIRenderer> Ptr;
    typedef std::vector<Ptr> Ptrs;

    UIRenderer() : m_id(s_newId++), Enabled(true), Visible(true), Rectangle(0, 0, 128, 16), AlphaBlending(1.0f), AlignToLeft(true), AlignToTop(true) {}
    virtual ~UIRenderer() {}
    inline void AddChild(const Ptr& child) { m_children.push_back(child); }
    bool Render(cv::Mat& canvas, const cv::Point& origin = Origin);
    inline size_t GetId() const { return m_id; }
    bool     Enabled;
    bool     Visible;
    cv::Rect Rectangle;
    float    AlphaBlending;
    bool     AlignToLeft;
    bool     AlignToTop;
    virtual void OnResized() {}

protected:
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin) = 0;

    static const cv::Point Origin;
    Ptrs m_children;

private:
    static size_t s_newId;
    size_t m_id;
};

class Label : public UIRenderer
{
public:
    typedef boost::shared_ptr<Label> Ptr;
    typedef std::vector<Ptr> Ptrs;

    enum VerticalAlign   {Top, Middle, Bottom};
    enum HorizontalAlign {Left, Centre, Right};

    static Ptr Create(const String& caption) { return Label::Ptr(new Label(caption)); }

    Label(const String& caption);
    inline void SetBackColour(cv::Scalar& colour) { MakeDisabledColour(m_backColour = colour, m_disabledBackColour); }
    inline void SetForeColour(cv::Scalar& colour) { MakeDisabledColour(m_foreColour = colour, m_disabledForeColour); }
    inline cv::Scalar GetBackColour() const { return m_backColour; }
    inline cv::Scalar GetForeColour() const { return m_foreColour; }
    cv::Size GetTextSize() const;

    String Caption;
    size_t BorderWidth;
    int    FontFace;
    double FontScale;
    int    FontWeight;
    bool   Highlighted;
    bool   Filled;
    size_t Paddings;

    VerticalAlign   VerticalAlign;
    HorizontalAlign HorizontalAlign;

protected:
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin);

private:
    static void MakeDisabledColour(const cv::Scalar& enabled, cv::Scalar& disabled);

    cv::Scalar m_backColour;
    cv::Scalar m_foreColour;
    cv::Scalar m_disabledBackColour;
    cv::Scalar m_disabledForeColour;
};

class CascadedMenu : public UIRenderer
{
public:
    typedef boost::shared_ptr<CascadedMenu> Ptr;

    class MenuEventListener
    {
    public:
        virtual void OnItemChanged(const CascadedMenu& source, size_t itemIdx) = 0;
    };

    class MenuItem : public Label, public Indexed
    {
    public:
        typedef boost::shared_ptr<MenuItem> Ptr;
        typedef std::vector<Ptr> Ptrs;
    protected:
        friend class CascadedMenu;
        MenuItem(const String& caption) : Label(caption) {}
    };

    class Togglable : public MenuItem
    {
    public:
        typedef boost::shared_ptr<Togglable> Ptr;
    protected:
        friend class CascadedMenu;
        Togglable(const String& caption, bool toggled)
        : m_toggled(toggled), MenuItem(caption) {}
        inline bool IsToggled() const        { return m_toggled; }
        inline void SetToggled(bool toggled) { m_toggled = toggled; }
        inline bool Toggle()                 { return (m_toggled = !m_toggled); }
    private:
        bool m_toggled;
    };

    class Expandable : public MenuItem
    {
    public:
        typedef boost::shared_ptr<Expandable> Ptr;
        typedef std::vector<Ptr> Ptrs;
        CascadedMenu::Ptr GetSubmenu() { return m_submenu; }
    protected:
        friend class CascadedMenu;
        Expandable(const String& caption, size_t toggles)
        : m_submenu(new CascadedMenu(toggles)), MenuItem(caption), m_expanded(false) {}
        inline bool IsExpanded() const { return m_expanded; }
        inline void SetExpanded(bool expanded) { m_expanded = expanded; }
        inline bool CanExpand() const { return m_submenu->m_items.size() > m_submenu->m_toggles; }
    private:
        CascadedMenu::Ptr m_submenu;
        bool m_expanded;
    };

    static Ptr Create(size_t toggles = 0)
    { return Ptr(new CascadedMenu(toggles)); }

    Togglable ::Ptr CreateTogglableItem (const String& caption);
    Expandable::Ptr CreateExpandableItem(const String& caption, size_t toggles = 0);
    virtual void OnResized();
    void NextItem(int inc = 1);
    inline void PrevItem() { NextItem(-1); }
    bool NextLevel();
    bool PrevLevel();
    void SetListener(MenuEventListener* listener) { m_listener = listener; }
    inline const MenuItem::Ptr GetItem(size_t idx) const { return m_items[idx]; }
    inline std::vector<size_t> GetToggled() const { return m_toggled; }

protected:
    CascadedMenu(size_t toggles)
    : m_toggles(toggles), m_highlight(0), m_listener(NULL),
      m_darkColour1(0, 0, 32), m_darkColour2(0, 0, 255), m_lightColour(255, 255, 255) {}
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin);
    void AddItem(const MenuItem::Ptr& item);

private:
    static void DrawArrow(cv::Mat& canvas, const cv::Rect& rect, const cv::Scalar& colour);
    static const size_t s_itemHeight;
    static const size_t s_vspace;
    static const size_t s_hspace;

    void UpdateToggle(Togglable::Ptr togglable, bool toggled);

    MenuItem::Ptrs m_items;
    int            m_highlight;
    const size_t   m_toggles;
    std::vector<size_t> m_toggled;
    MenuEventListener* m_listener;

    cv::Scalar m_darkColour1;
    cv::Scalar m_darkColour2;
    cv::Scalar m_lightColour;
};

class BufferWriterStatsRenderer : public UIRenderer
{
public:
    BufferWriterStatsRenderer(const BufferWriter::Ptr& writer, size_t numRecords = 100);
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin);

private:
    typedef std::vector<double> BpsRecords;

    BufferWriter::Ptr m_writer;
    BpsRecords m_records;
    size_t     m_currentIdx;
    size_t     m_bestIdx;
    Label::Ptr m_grabberLabel;
    Label::Ptr m_bpsLabel;
    Label::Ptr m_seqLabel;
    Label::Ptr m_fpsLabel;
    Label::Ptr m_delayLabel;
    cv::Rect   m_plotRect;

};

class BufferUsageIndicator : public UIRenderer
{
public:
    BufferUsageIndicator(const SyncBuffer::Ptr& buffer);
    virtual void OnResized();
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin);
private:
    SyncBuffer::Ptr m_buffer;
    Label::Ptr      m_usageLabel;
    cv::Rect        m_barsRect;
};

class RecorderStatsRenderer : public UIRenderer
{
public:
    RecorderStatsRenderer(BufferRecorder& rec);
    virtual void OnResized();
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin);
private:
    BufferRecorder& m_rec;
    size_t          m_seq;
    Label::Ptr      m_recordingLabel;
    Label::Ptr      m_miscsLabel;
};

class MainUI : public UIRenderer, protected CascadedMenu::MenuEventListener
{
public:
    MainUI(const ImageGrabber::Ptrs& grabbers, BufferRecorder& recorder, const SyncBuffer::Ptr& buffer);
    virtual ~MainUI() {}
    void Loop();

protected:
    virtual bool Draw(cv::Mat& canvas, const cv::Point& origin);
    virtual void OnItemChanged(const CascadedMenu& source, size_t itemIdx);

private:
    static const String s_caption;
    static const int    s_waitKeyTimeout;

    void KeyPressed(int key);

    ImageGrabber::Ptrs m_grabbers;
    BufferRecorder&    m_recorder;
    SyncBuffer::Ptr    m_buffer;
    CascadedMenu::Ptr  m_menu;
    CascadedMenu::Ptr  m_activeViewMenu;
    Ptr                m_bufUsageIndicator;
    Ptr                m_recorderIndicator;
    size_t             m_vmMenuId;
    size_t             m_smMenuId;
    size_t             m_vmF3idx;
    size_t             m_vmF2idx;
    size_t             m_vmS3idx;
    size_t             m_vmS2idx;
    size_t             m_vmS1idx;
    bool               m_mixchan;
    uchar              m_flashing;
};

#endif // UIRENDERER_HPP
