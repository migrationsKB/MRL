from pylode.ontdoc import OntDoc


def main():
    f = 'populate_kb/input/mgkb_schema.ttl'
    output = "populate_kb/input/mgkb_schema.html"
    print('making ontdoc for {}'.format(f))
    OntDoc(f).make_html(destination=output)


if __name__ == '__main__':
    main()
