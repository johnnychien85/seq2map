#ifndef SPAMAP_HPP
#define SPAMAP_HPP

#include <boost/enable_shared_from_this.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <seq2map/common.hpp>

namespace seq2map
{
namespace spamap
{
    template<typename T>
    class Node : public boost::enable_shared_from_this<Node<T> >
    {
    public:
        class Row;
        class Column;
        typedef Column Col;

        virtual ~Node()
        {
            // Make sure the Node has been fully detached when it's being destroyed.
            // Node deletion should only be invoked automatically by smart pointers.
            assert(up.expired());
            assert(down.expired());
            assert(left.expired());
            assert(!right);

            //E_TRACE << ToString() << " destroyed";
        }

        inline String ToString() const
        {
            std::stringstream ss;
            ss << "(" << row.GetIndex() << "," << col.GetIndex() << ")";

            return ss.str();
        }

    private:
        typedef Node<T> NodeType;
        typedef boost::shared_ptr<NodeType> Own;
        typedef boost::weak_ptr  <NodeType> Ref;

        // can only be constructed by Row::Insert
        Node(Row& row, Col& col, const T& value) : row(row), col(col), value(value), indexed(false)
        {
            //E_TRACE << ToString() << " borned";
        }

    public:
        Row& row;
        Col& col;
        T value;

        Ref up;
        Ref down;
        Ref left;
        Own right;

        bool indexed;

    public:
        class Header : public Indexed
        {
        public:
            Header(size_t idx) : Indexed(idx) {}
            virtual bool empty() const { return !m_head; }
            virtual void clear() = 0;
        
        protected:
            Own m_head;
            Own m_tail;

        public: // STL-compliance members
            typedef T value_type;
        };

        class Iterator
        : public boost::iterator_facade<Iterator, NodeType, boost::bidirectional_traversal_tag, T&>
        {
        public:
            Iterator() {}
            explicit Iterator(Own& node) : m_node(node) {}

            bool operator==(Iterator const& itr) const
            {
                return equal(itr);
            }

        protected:
            friend class boost::iterator_core_access;

            bool equal(Iterator const& itr) const
            {
                return m_node == itr.m_node;
            }

            T& dereference() const { return m_node->value; }

            virtual void increment() = 0;
            virtual void decreasement() = 0;

            Own m_node;
        };

        class Row : public Header
        {
        public:
            Row(size_t idx = INVALID_IDX) : Header(idx) {}
            ~Row() { clear(); }

            T& Insert(Col& col, const T& value)
            {
                Own node = Own(new NodeType(*this, col, value));
                Own prev = m_tail;
                Own next;

                if (empty())
                {
                    assert(!m_head);
                    m_head = m_tail = node->shared_from_this();
                }
                else
                {
                    while (prev && node->col < prev->col)
                    {
                        next = prev;
                        prev = prev->left.lock();
                    }

                    if (next) // take over the ownership
                    {
                        node->right = next;
                        next->left  = node->shared_from_this();

                        // E_TRACE << node->ToString() << " <----> " << next->ToString();
                    }
                    else
                    {
                        m_tail = node->shared_from_this();
                    }

                    if (prev)
                    {
                        node->left  = prev;
                        prev->right = node->shared_from_this();

                        // E_TRACE << prev->ToString() << " <----> " << node->ToString();
                    }
                    else
                    {
                        m_head = node->shared_from_this();
                    }
                }

                col.Insert(node, prev);

                return node->value;
            }

            void Remove(Own node)
            {
                assert(node);
                assert(&node->row == this); // ownership check

                node->col.Remove(node);

                Own prev = node->left.lock();
                Own next = node->right;

                if (prev)
                {
                    prev->right = next;
                    // E_TRACE << prev->ToString() << " <-//-> " << node->ToString();
                }
                else
                {
                    assert(m_head == node);
                    m_head = next;
                }

                if (next)
                {
                    next->left = prev;
                    // E_TRACE << node->ToString() << " <-//-> " << next->ToString();
                }
                else
                {
                    assert(m_tail == node);
                    m_tail = prev;
                }

                // detach the node completely
                node->up = node->down = node->left = node->right = Own();
            }

        public: // STL-compliance members
            class Iterator : public NodeType::Iterator
            {
            public:
                Iterator(Own& node = Own()) : NodeType::Iterator(node) {}
            protected:
                virtual void increment() { m_node = m_node->right; }
                virtual void decreasement() { m_node = m_node->left.lock(); }
            };

            typedef Iterator iterator;

            inline iterator begin() { return Iterator(m_head); }
            inline iterator end()   { return Iterator(); }

            virtual inline void clear()
            {
                while (m_tail) Remove(m_tail);
                assert(empty());
            }
        };

        class Column : public Header
        {
        public:
            Column(size_t idx = INVALID_INDEX) : Header(idx) {}
            ~Column() { clear(); }

        protected:
            friend class Row;

            void Insert(Own& node, Own& hint)
            {
                assert(&node->col == this); // ownership check

                const size_t row = node->row.GetIndex();
                const size_t col = node->col.GetIndex();

                Own prev;
                Own next;

                // check if the hint can be used or not
                bool hit = hint && hint->col.GetIndex() == col;

                if (hit)
                {
                    // structural checks
                    assert(hint->row.GetIndex() == row);
                    assert(!empty());

                    prev = hint;
                    next = hint->down.lock(); // could be null
                }
                else if(!empty()) // find the predecessor and successor if there's any
                {
                    // find the first element whose row is later than the new node
                    NodeRefs::iterator itr = m_nodes.lower_bound(row);

                    if (itr != m_nodes.end()) // got a successor
                    {
                        next = itr->second.lock(); // the successor
                        prev = next ? next->up.lock() : Own(); // the predeccessor, could be null

                        // neither successor nor predeccessor (if exists) can be at the sam
                        // or it should have been detected and passed as a hint by Row::Insert()
                        assert( next && next->row.GetIndex() != row);
                        assert(!prev || prev->row.GetIndex() != row);
                    }
                    else // insert to the end of column
                    {
                        itr--; // move to the last element

                        prev = itr->second.lock(); // the predeccessor
                        next = prev ? prev->down.lock() : Own(); // the successor, has to be null

                        assert( prev && prev->row.GetIndex() != row); // see the other case
                        assert(!next); // the end of column has nothing down there
                    }
                }

                if (prev)
                {
                    prev->down = node->shared_from_this();
                    node->up   = prev;

                    //E_TRACE << prev->ToString() << " <----> " << node->ToString();
                }
                else
                {
                    m_head = node->shared_from_this();
                }

                if (next)
                {
                    next->up   = node->shared_from_this();
                    node->down = next;

                    //E_TRACE << node->ToString() << " <----> " << next->ToString();
                }
                else
                {
                    m_tail = node->shared_from_this();
                }

                if (!hit) // update the index for the first hit
                {
                    m_nodes[row] = node->shared_from_this();
                    node->indexed = true;
                }
            }

            void Remove(Own& node)
            {
                assert(node);
                assert(&node->col == this);

                const size_t row = node->row.GetIndex();

                Own prev = node->up.lock();
                Own next = node->down.lock();

                if (node->indexed)
                {
                    Own newIndex;

                    if      (prev && prev->row.GetIndex() == row) newIndex = prev;
                    else if (next && next->row.GetIndex() == row) newIndex = next;

                    if (newIndex)
                    {
                        newIndex->indexed = true;
                        m_nodes[row] = newIndex;
                    }
                    else
                    {
                        size_t erased = m_nodes.erase(row);
                        assert(erased == 1);
                    }

                    node->indexed = false;
                }

                if (prev)
                {
                    prev->down = next;
                    // E_TRACE << prev->ToString() << " <-//-> " << node->ToString();
                }
                else
                {
                    m_head = next;
                }

                if (next)
                {
                    next->up = prev;
                    // E_TRACE << node->ToString() << " <-//-> " << next->ToString();
                }
                else
                {
                    m_tail = prev;
                }
            }

        private:
            typedef std::map<size_t, Ref> NodeRefs;
            NodeRefs m_nodes;

        public: // STL-compliance members
            class Iterator : public NodeType::Iterator
            {
            public:
                Iterator(Own& node = Own()) : NodeType::Iterator(node) {}
            protected:
                virtual void increment() { m_node = m_node->down.lock(); }
                virtual void decreasement() { m_node = m_node->up.lock(); }
            };

            typedef Iterator iterator;

            inline iterator begin() { return Iterator(m_head); }
            inline iterator end()   { return Iterator(); }

            virtual inline void clear()
            {
                while (m_tail) m_tail->row.Remove(m_tail);

                // the indexing tree should be empty now
                assert(m_nodes.empty());
            }
        };
    };

    template<typename T, class R = Node<T>::Row, class C = Node<T>::Col>
    class Map
    {
    public:
        typedef R RowType;
        typedef C ColType;

        Map() {}

        virtual ~Map()
        {
            m_cols.clear(); // remove columns first
            m_rows.clear(); // then rows..
        }

        R& Row(size_t i)
        {
            Rows::iterator itr = m_rows.find(i);
            return itr != m_rows.end() ? itr->second :
                m_rows.insert(Rows::value_type(i, R(i))).first->second;
        }

        C& Col(size_t j)
        {
            Cols::iterator itr = m_cols.find(j);
            return itr != m_cols.end() ? itr->second :
                m_cols.insert(Cols::value_type(j, C(j))).first->second;
        }

        inline T& Insert(size_t i, size_t j, const T& value)
        {
            return Row(i).Insert(Col(j), value);
        }

        //inline T& operator(size_t i, size_t j)
        //{
        //    return Insert(i, j, T());
        //}

    private:
        typedef std::map<size_t, R> Rows;
        typedef std::map<size_t, C> Cols;

        Rows m_rows;
        Cols m_cols;
    };
}
}
#endif // SPAMAP_HPP
